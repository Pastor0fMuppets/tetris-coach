"""Shape recognition: falling piece in the board grid, next piece in a preview image.

Both work purely on shape (occupancy); color is never required.

The falling piece is never guessed geometrically from a single frame.
Instead, :func:`explain_grid` diffs the observed board against the
tracker's committed stack memory and classifies the frame:

- cells appearing on top of an intact stack are the falling piece
  (wherever they are — resting on the stack during lock delay included);
- a lock is recognized only when a structurally verified transition
  explains the whole frame (the next spawn revealing it, or a line-clear
  placement reproducing the observation exactly);
- anything else is UNEXPLAINED and produces no candidate at all.

Some cells of the captured board may be *unobservable* — permanently
covered by a game's own UI (the ROAS Stacker NEXT preview floats over the
top corner of the playfield; see :func:`~tetris_coach.app.compute_overlap_mask`).
They are passed in as ``unknown_rows`` and mean "no evidence", never
"empty": they are excluded from both sides of the diff, and they may stand
in for whatever cells a hypothesis needs (the hidden half of a piece, the
hidden half of a lock). A frame whose visible evidence is only the
visible part of a piece is :attr:`FrameKind.OCCLUDED` — explained, but
carrying no candidate, so it holds state instead of tripping a reset.

The board region has one unobservable region no mask can name: the space
ABOVE row 0, which pieces enter the playfield through. A piece that has
spawned but not yet descended is cut by the capture's top edge and shows
only its bottom 1-3 cells, and in a game without gravity (drag to move,
drag to drop) it SITS there for seconds rather than falling through in a
frame. Those cells are read exactly like cells at the edge of a panel:
named when one tetromino completes them, OCCLUDED when several do — and
never committed as stack content, which is the corruption that made every
hint after the first one wrong (see :func:`_clipped_completions`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
from numpy.typing import NDArray

from ..core.board import FULL_ROW, WIDTH, Board
from ..core.pieces import PIECES, ROTATIONS
from .grid import (
    _GHOST_SEPARATION,
    HINT_PAINT,
    MIN_SPREAD,
    OwnPaint,
    _distance_scores,
    otsu_threshold_hist,
)

Cell = tuple[int, int]

# All standard games spawn pieces within the top rows of the visible field;
# a spawn candidate's bounding box must start above this row.
SPAWN_ROWS = 4

# Normalized cell set -> (piece, rotation index), for O(1) shape matching.
_SHAPE_LOOKUP: dict[tuple[Cell, ...], tuple[str, int]] = {
    rot.cells: (rot.piece, rot.index) for rots in ROTATIONS.values() for rot in rots
}


@dataclass(frozen=True)
class FallingPiece:
    """A recognized falling piece, positioned on the board grid."""

    piece: str
    rotation_index: int
    # Top row of the bounding box on the board. NEGATIVE while the piece is
    # still entering from above the board region: the bounding box starts
    # off-grid and :attr:`cells` then names cells at rows < 0, which no
    # board-indexed rule may touch (they are not board cells at all).
    row: int
    col: int  # left column of the bounding box on the board

    @property
    def cells(self) -> tuple[Cell, ...]:
        """Absolute (row, col) board cells occupied by the piece."""
        return tuple(
            (self.row + r, self.col + c)
            for r, c in ROTATIONS[self.piece][self.rotation_index].cells
        )


class FrameKind(Enum):
    """Structural classification of one observed frame."""

    QUIET = auto()  # nothing new on the board (post-lock, pre-spawn)
    FALLING = auto()  # stack intact + exactly one tetromino of new cells
    LOCKED = auto()  # a lock (with or without clears) exactly verified
    OCCLUDED = auto()  # a piece seen in part: under a panel, or cut by the top edge
    UNEXPLAINED = auto()  # no structural explanation; produce no candidate


@dataclass(frozen=True)
class Explanation:
    """Outcome of :func:`explain_grid` for one frame."""

    kind: FrameKind
    stack_rows: tuple[int, ...]  # meaningful for QUIET/FALLING/LOCKED only
    falling: FallingPiece | None  # position included (raw observation)
    # True when ``falling``'s NAME came from the entering hint rather than
    # from the frame's own structure: the preview said which piece is
    # entering, and several tetrominoes fit the cells on screen. A guess
    # backed by evidence is good enough to hint on, and not good enough to
    # become evidence itself (see
    # :meth:`~tetris_coach.vision.state.GameStateTracker.update`).
    hinted_name: bool = False
    # True when the cells on screen RULE OUT the entering hint: the
    # fragment is a piece coming in from above, and no placement of the
    # hinted piece fits it. The preview named a hypothesis and the piece
    # has now shown enough of itself to contradict it, so the tracker
    # drops the hypothesis and goes back to naming from shape alone (see
    # :meth:`~tetris_coach.vision.state.GameStateTracker.update`).
    hint_refuted: bool = False
    # Every piece that could be the one entering from above, given this
    # frame's fragment — empty when the frame shows no entering fragment
    # at all. ``hint_refuted`` is this set tested against the hint the
    # tracker passed IN; the set itself is here so the tracker can run the
    # same test against a name it has already COMMITTED from a hint, on
    # frames where there is no live hint left to refute (see
    # :meth:`~tetris_coach.vision.state.GameStateTracker.update`).
    entering_names: frozenset[str] = frozenset()


def match_cells(cells: frozenset[Cell] | set[Cell] | tuple[Cell, ...]) -> tuple[str, int] | None:
    """Match a set of 4 absolute cells against every tetromino rotation.

    Returns ``(piece, rotation_index)`` or ``None``.
    """
    cell_list = list(cells)
    if len(cell_list) != 4:
        return None
    min_r = min(r for r, _ in cell_list)
    min_c = min(c for _, c in cell_list)
    normalized = tuple(sorted((r - min_r, c - min_c) for r, c in cell_list))
    return _SHAPE_LOOKUP.get(normalized)


def _connected_components(cells: Iterable[Cell]) -> list[set[Cell]]:
    """4-connected components of a cell set."""
    remaining = set(cells)
    components: list[set[Cell]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = {seed}
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _cells_from_rows(rows: Iterable[int]) -> set[Cell]:
    """All (row, col) cells set in an iterable of row bitmasks."""
    cells: set[Cell] = set()
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            row ^= bit
            cells.add((r, bit.bit_length() - 1))
    return cells


def _rows_with(rows: tuple[int, ...], cells: Iterable[Cell]) -> tuple[int, ...]:
    """``rows`` with ``cells`` merged in."""
    out = list(rows)
    for r, c in cells:
        out[r] |= 1 << c
    return tuple(out)


def _piece_at(cells: Iterable[Cell]) -> FallingPiece | None:
    """Match a cell set against every tetromino, positioned on the board."""
    cell_tuple = tuple(cells)
    matched = match_cells(cell_tuple)
    if matched is None:
        return None
    piece, rotation_index = matched
    return FallingPiece(
        piece=piece,
        rotation_index=rotation_index,
        row=min(r for r, _ in cell_tuple),
        col=min(c for _, c in cell_tuple),
    )


def _bit(rows: tuple[int, ...], cell: Cell) -> bool:
    """True when ``cell`` is set in a row-bitmask tuple."""
    r, c = cell
    return bool(rows[r] >> c & 1)


def _hidden_completions(
    added: set[Cell],
    unknown_rows: tuple[int, ...],
    height: int,
) -> set[frozenset[Cell]]:
    """Tetromino placements whose *visible* new cells are exactly ``added``.

    A piece straddling the boundary of an unobservable region shows only
    part of itself. This enumerates every placement that contains all of
    ``added`` and whose remaining cells all fall inside ``unknown_rows`` —
    i.e. every piece the frame could be hiding. No stack test is needed:
    ``added`` cells are new by construction and the hidden cells are
    unobservable, where the committed stack holds a belief, not evidence,
    and so cannot rule a placement out.

    Every completion contains every added cell, so anchoring the search on
    one of them enumerates all of them.
    """
    anchor_r, anchor_c = next(iter(added))
    completions: set[frozenset[Cell]] = set()
    for rots in ROTATIONS.values():
        for rot in rots:
            for cell_r, cell_c in rot.cells:
                top, left = anchor_r - cell_r, anchor_c - cell_c
                if top < 0 or left < 0 or top + rot.height > height or left + rot.width > WIDTH:
                    continue
                cells = frozenset((top + r, left + c) for r, c in rot.cells)
                if added <= cells and all(_bit(unknown_rows, cell) for cell in cells - added):
                    completions.add(cells)
    return completions


def _clipped_completions(
    fragment: set[Cell],
    stack_rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
) -> set[FallingPiece]:
    """Pieces whose visible part is exactly ``fragment``, cut by the TOP EDGE.

    Pieces enter the playfield from above the board region, so the capture's
    top edge cuts a just-spawned piece in half and 1-3 of its cells are on
    the grid. Enumerates every placement whose bounding box starts above
    row 0 and whose on-board cells are ``fragment`` (plus, as everywhere,
    cells inside ``unknown_rows``, which are evidence for nothing).

    Two guards keep real board content out of this rule:

    - the fragment must touch row 0. A piece the top edge cuts always
      reaches it; a blob floating at row 3 is something else entirely.
    - the fragment must be AIRBORNE (``not _supported``). A piece entering
      from above has air under it; cells at row 0 resting on the stack are
      the top of the stack — a lock near top-out, or a row about to clear —
      and reading those as an entering piece would delete them.

    The returned pieces carry a negative ``row`` (the bounding box starts
    off-grid). An empty set means "not a piece entering from above" and the
    caller falls through to the ordinary rules unchanged.
    """
    if not 1 <= len(fragment) <= 3:
        return set()
    if not any(r == 0 for r, _ in fragment):
        return set()
    if _supported(fragment, stack_rows):
        return set()
    height = len(stack_rows)
    anchor_r, anchor_c = next(iter(fragment))
    completions: set[FallingPiece] = set()
    for rots in ROTATIONS.values():
        for rot in rots:
            for cell_r, cell_c in rot.cells:
                top, left = anchor_r - cell_r, anchor_c - cell_c
                if top >= 0:
                    continue  # fully on the board: the ordinary rules own it
                if left < 0 or left + rot.width > WIDTH or top + rot.height > height:
                    continue
                cells = {(top + r, left + c) for r, c in rot.cells}
                visible = {cell for cell in cells if cell[0] >= 0}
                if not fragment <= visible:
                    continue
                if not all(_bit(unknown_rows, cell) for cell in visible - fragment):
                    continue
                completions.add(FallingPiece(rot.piece, rot.index, top, left))
    return completions


def _partial_completions(
    fragment: set[Cell],
    stack_rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
) -> set[FallingPiece]:
    """Every piece whose VISIBLE new cells are exactly ``fragment``.

    A fragment of 1-3 cells is a piece the capture cannot see whole, and
    there are two ways for that to happen at once: cells hidden under a UI
    panel (:func:`_hidden_completions`) and cells above the board region's
    top edge (:func:`_clipped_completions`). Both are hypotheses about the
    SAME cells, so they are one candidate set and not two.

    Asking the panel first and answering from it alone is what this
    replaces: in the live session's own geometry (a panel at rows 0-1,
    cols 8-9) the fragment ``{(0,5), (0,6), (0,7)}`` has exactly ONE panel
    completion — a horizontal I whose fourth cell hides under the panel —
    and three clipped ones (a T, a J and an L entering from above). The
    panel's answer is a guess dressed as a unique completion, and it comes
    with a position as well as a name.
    """
    completions: set[FallingPiece] = {
        piece
        for cells in _hidden_completions(fragment, unknown_rows, len(stack_rows))
        if (piece := _piece_at(cells)) is not None
    }
    return completions | _clipped_completions(fragment, stack_rows, unknown_rows)


@dataclass(frozen=True)
class PartialRead:
    """What :func:`_partial_piece` made of a 1-3 cell fragment."""

    coherent: bool  # some piece is there, whether or not it can be named
    piece: FallingPiece | None  # named, when exactly one candidate fits
    hinted: bool  # the name came from the entering hint, not from shape
    refuted: bool  # the cells rule the entering hint OUT
    entering_names: frozenset[str] = frozenset()  # every name still open


def _partial_piece(
    fragment: set[Cell],
    stack_rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
    entering_hint: str | None = None,
) -> PartialRead:
    """Read a 1-3 cell fragment: is it a partly seen piece, and which one?

    Ambiguity is reported as coherent with no piece: the frame makes
    sense — something is there — but two horizontally adjacent cells at
    row 0 fit an O, an S, a Z, a J and an L alike, and a guessed name is a
    guessed hint. Holding beats guessing; the piece names itself as soon as
    it is seen whole.

    ``entering_hint`` breaks that tie when there is EVIDENCE for the name
    rather than a guess: the piece that just left the NEXT preview is the
    piece now entering the board (the tracker watches the preview change
    and passes the departing name; see
    :meth:`~tetris_coach.vision.state.GameStateTracker.update`). It only
    ever SELECTS among completions the structural rule already accepts —
    every guard still holds — and only when exactly one candidate on the
    whole frame carries that name AND that candidate is one entering from
    above, which is the only thing the hint is evidence about: a piece
    sliding under a panel was named from its own earlier frames. A
    mis-read preview can therefore misname an entering piece, and that is
    the deliberate trade: the alternative is no name for as long as the
    piece sits at the top edge (measured on the live session: 13 frames,
    ~0.9 s, of a piece nobody could hint), and a misnamed one stands only
    until the frame can contradict it — which is as long as the fragment
    stays ambiguous, 12 frames on the committed window when the box lies
    consistently, and nothing structural either way.
    ``hinted`` says the name came from the hint rather than from
    structure, so the tracker can refuse to treat a guess as an
    observation.

    ``refuted`` says the cells RULE THE HINT OUT: they are a piece
    entering from above and no placement of the hinted piece fits them. A
    preview reading is a hypothesis, and this is the frame that falsifies
    it — a piece two cells wide fits an O as well as a T, but its next row
    down does not, so a misnamed piece contradicts itself within a frame
    or two of descending. The hypothesis is then dropped rather than
    carried to the end of the deal, and the piece names itself from shape
    as usual. Refuted and named are not exclusive: the fragment can rule
    the hint out and have exactly one completion of its own.

    ``entering_names`` is that test's raw material — every piece the
    fragment still leaves open as the one entering from above, empty when
    nothing is entering. The tracker runs the same test against a name it
    has already COMMITTED from a hint, which is a different subject from
    the live hint: the hint is dropped at the first contradicting frame,
    and the name it gave the piece on screen has to be taken back on that
    frame too, or a refuted guess stays on the overlay.
    """
    completions = _partial_completions(fragment, stack_rows, unknown_rows)
    if not completions:
        return PartialRead(False, None, False, False)
    entering = [piece for piece in completions if piece.row < 0]
    # Only a fragment coming in from above is evidence about the hint: a
    # piece sliding under a UI panel was named from its own earlier
    # frames, and the hint says nothing about it either way.
    open_names = frozenset(piece.piece for piece in entering)
    refuted = entering_hint is not None and bool(entering) and entering_hint not in open_names
    if len(completions) == 1:
        return PartialRead(True, next(iter(completions)), False, refuted, open_names)
    if entering_hint is not None:
        hinted = [piece for piece in completions if piece.piece == entering_hint]
        if len(hinted) == 1 and hinted[0].row < 0:
            return PartialRead(True, hinted[0], True, False, open_names)
    return PartialRead(True, None, False, refuted, open_names)


# The most content a single frame is ever allowed to hold back as "in
# flight": one tetromino. Nothing bigger is a piece, and past that budget
# keeping the content beats inventing a reason to delete it.
MAX_IN_FLIGHT_CELLS = 4


def _rows_without(rows: tuple[int, ...], cells: Iterable[Cell]) -> tuple[int, ...]:
    """``rows`` with ``cells`` removed."""
    out = list(rows)
    for r, c in cells:
        out[r] &= ~(1 << c)
    return tuple(out)


def _resting(
    component: set[Cell],
    rest_rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
) -> bool:
    """True when ``component`` may be sitting on something.

    ``rest_rows`` is the board WITHOUT the component, so a cell of the
    component itself never props it up. An unobservable cell directly
    underneath counts as support: what the capture cannot read is evidence
    for nothing, and treating it as air would let this rule delete content
    on a guess. Everything here errs toward "resting", i.e. toward leaving
    the content where it is.
    """
    if _supported(component, rest_rows):
        return True
    return any(r + 1 < len(rest_rows) and _bit(unknown_rows, (r + 1, c)) for r, c in component)


def _touches_unobservable(component: set[Cell], unknown_rows: tuple[int, ...]) -> bool:
    """True when any cell of ``component`` is 4-adjacent to a covered cell."""
    height = len(unknown_rows)
    for r, c in component:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < WIDTH and _bit(unknown_rows, (nr, nc)):
                return True
    return False


def _floating_component(
    rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
) -> set[Cell] | None:
    """The one component of ``rows`` that rests on nothing, if there is one.

    Components are MAXIMAL, so the only things that can be directly under
    one are the floor, an unobservable cell, or air. A component with air
    under all of it is therefore floating — and settled stack cells do not
    float, whatever else they do (overhangs and covered holes are all part
    of a component that reaches the floor).

    ``None`` when nothing floats, or when two things do: two floating blobs
    name no single piece, and absorbing both keeps content, which is the
    safe direction.

    "Maximal" holds only on the board the capture can SEE. An unobservable
    cell is blanked before any of this runs, so a component that touches
    one may be a piece of a larger component the panel cuts in half, and
    the half that reaches the floor can be on the other side: in this
    session's own geometry (the NEXT preview over cols 8-9 of rows 0-1) a
    column stacked to the top with a piece locked beside it at cols 6-7 is
    ONE grounded component in the true board and two components here, the
    inner one floating. Support directly below is not the only way through
    — :func:`_resting` answers that one — so a component ADJACENT to an
    unobservable cell in any direction is not demonstrably floating, and
    the resync absorbs it like any other content it cannot rule on.
    """
    floating = [
        component
        for component in _connected_components(_cells_from_rows(rows))
        if not _resting(component, _rows_without(rows, component), unknown_rows)
        and not _touches_unobservable(component, unknown_rows)
    ]
    if len(floating) != 1:
        return None
    return floating[0]


def strip_piece_in_flight(
    rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
) -> tuple[int, ...]:
    """``rows`` without the piece in flight, whole or clipped.

    For the one path that adopts a board with no memory to diff it against
    (the tracker's resync). Everything on such a frame is taken to be the
    stack, because there is no evidence for anything else — except content
    that is demonstrably NOT settled: a single component with air under all
    of it. Stack cells do not float, so a floating component is the falling
    piece, and freezing it into the stack is the corruption that makes the
    piece's own descent look like stack cells vanishing — which nothing can
    explain, so the tracker resets again and re-absorbs it one row lower,
    the whole way down (see ``tests/fixtures/absorbed_piece``).

    Budgeted at ONE tetromino, and conservative in every direction past
    that. Held back only when:

    - exactly one component floats (two at once name no single piece);
    - it is at most :data:`MAX_IN_FLIGHT_CELLS` cells;
    - at exactly four cells it is a tetromino — a four-cell blob that is no
      piece is something else and stays;
    - and it TOUCHES ROW 0, i.e. it is entering or has just spawned.

    That last one is the load-bearing one, because "floating" does NOT
    imply "in flight". Naive gravity makes SETTLED cells float: clear a
    row and everything above descends onto a row with holes in it, so the
    repo's own :meth:`Board.drop` turns an ordinary position (stack at
    rows 8-9 of cols 0-1 over holes at row 11) into a board with a real
    locked O floating at rows 9-10. Without the row-0 test a resync
    deletes those four settled cells; the next frames read the same cells
    as added, match an O, and the tracker announces a piece the player
    does not have on a board four cells short of the truth. The same
    premise failure with a 1-3 cell post-clear remnant was worse than
    one-shot: the shortened board equals the committed one, the resync
    no-ops, and the tracker never adopts the real content at all
    (measured: 300 identical frames, zero events, three real cells
    missing for good).

    Settled content cannot be at row 0 with air under it — that is the
    one arrangement the board itself rules out, and it is the picture
    ``grid.py`` already vouches for as "one piece in flight" when it
    judges confidence. Anything lower is the same picture as post-clear
    debris, and the costs are not symmetric, so it stays. A piece that
    was absorbed lower down is not stranded: its next descending frame
    takes it back out (:func:`_carried_piece`), which is the loop-breaker
    that works however the piece got in. Measured over the three
    committed session replays and the 707-frame session: all 10 resyncs
    that hold anything back hold back a component touching row 0, so the
    test costs nothing on real frames.

    A component resting on the floor, on the rest of the board, or on an
    unobservable cell is absorbed exactly as it always was: a column
    stacked to the top, rising garbage and a piece in lock delay are all
    board content, and deleting any of them costs the solver a real block.
    Fewer than four floating cells need no shape test — a piece read in
    part (cut by the board region's top edge, sliding under a panel, or
    with a pale cell the threshold missed) is still a piece in flight, and
    1-3 floating cells fit inside some tetromino whatever they are.
    """
    fragment = _floating_component(rows, unknown_rows)
    if fragment is None or len(fragment) > MAX_IN_FLIGHT_CELLS:
        return rows
    if len(fragment) == MAX_IN_FLIGHT_CELLS and match_cells(fragment) is None:
        return rows
    if not any(r == 0 for r, _ in fragment):
        return rows  # floating lower down is post-clear debris, not evidence
    return _rows_without(rows, fragment)


def clear_full_rows(rows: tuple[int, ...]) -> tuple[int, ...]:
    """Remove full rows and prepend that many empty rows.

    Must match :meth:`Board.drop`'s clear step exactly (asserted in tests):
    the app compares boards produced by both paths.
    """
    kept = [row for row in rows if row != FULL_ROW]
    return tuple([0] * (len(rows) - len(kept)) + kept)


def _supported(cells: Iterable[Cell], stack_rows: tuple[int, ...]) -> bool:
    """True when the component rests on the floor or the stack.

    A component is supported when it could not descend one more row as a
    whole: some cell sits on the bottom row or directly above a stack cell.
    (Component cells are disjoint from the stack by construction, so a
    same-component cell below never counts as support.)
    """
    return any(r + 1 >= len(stack_rows) or bool(stack_rows[r + 1] >> c & 1) for r, c in cells)


def _explains(
    observed_rows: tuple[int, ...], s2_rows: tuple[int, ...], unknown_rows: tuple[int, ...]
) -> tuple[bool, FallingPiece | None]:
    """Verify a candidate post-lock stack ``s2_rows`` against the observation.

    Fails if any settled cell vanished. The residual (observed cells not in
    ``s2_rows``) must be empty, or exactly one tetromino spawning in the top
    rows — tolerating the next piece having already spawned during a clear.
    Unobservable cells are evidence for neither side and drop out of both
    comparisons.
    """
    if any(s & ~o & ~u for o, s, u in zip(observed_rows, s2_rows, unknown_rows, strict=True)):
        return False, None
    residual = _cells_from_rows(
        o & ~s & ~u for o, s, u in zip(observed_rows, s2_rows, unknown_rows, strict=True)
    )
    if not residual:
        return True, None
    piece = _piece_at(residual)
    if piece is not None and piece.row < SPAWN_ROWS:
        return True, piece
    # The spawn may still be entering from above the board region, showing
    # only its bottom cells: a clearing lock must not go UNEXPLAINED (and
    # eventually reset the board) because the next piece is half off-grid.
    read = _partial_piece(residual, s2_rows, unknown_rows)
    if read.coherent:
        return True, read.piece
    return False, None


def _lock_reveal(
    added: set[Cell],
    stack_rows: tuple[int, ...],
    last_falling: FallingPiece | None,
    unknown_rows: tuple[int, ...],
) -> Explanation | None:
    """A lock revealed by the next spawn, no clears (8 added cells).

    A candidate whose merged stack contains a full row is rejected: the
    game is mid clear animation (a zero-ARE flash frame still lights the
    completed row while the next piece is already visible), and that row
    is about to vanish. The frame stays UNEXPLAINED; the settled
    post-clear frame is explained by :func:`_lock_with_clears` instead.

    With unobservable cells in play the reveal carries fewer than 8 added
    cells (the locked piece's hidden cells never appear): L1 matches on the
    piece's *visible* cells and merges all of them, so the lock's hidden
    half enters the committed stack as the hypothesis says it must.
    """
    # L1 (position-anchored): the piece locked exactly where it was last
    # observed and the spawn appeared — even 4-adjacent to it (one blob).
    # A piece last seen half above the board region (row < 0) is skipped:
    # its cells are not all board cells, and a piece cannot lock up there.
    if last_falling is not None and last_falling.row >= 0:
        last_cells = set(last_falling.cells)
        visible_last = {cell for cell in last_cells if not _bit(unknown_rows, cell)}
        if visible_last <= added:
            spawn = _piece_at(added - last_cells)
            if spawn is not None and spawn.row < SPAWN_ROWS:
                merged = _rows_with(stack_rows, last_cells)
                if any(row == FULL_ROW for row in merged):
                    return None  # clear-flash frame: the full row will vanish
                return Explanation(FrameKind.LOCKED, merged, spawn)
    # L2 (structural): the hard-drop / zero-ARE case — the locked cells are
    # not the last observed cells, so split the diff into two tetrominoes.
    components = _connected_components(added)
    if len(components) != 2:
        return None
    valid: list[tuple[set[Cell], FallingPiece | None]] = []
    for lock_cells, spawn_cells in (
        (components[0], components[1]),
        (components[1], components[0]),
    ):
        spawn_piece = _piece_at(spawn_cells)
        if spawn_piece is None:
            # The spawn revealing the lock may be seen only in part: cut by
            # the top edge of the board region (this game spawns a piece the
            # moment the previous one is dropped, and the fragment SITS
            # there), or half under a panel. The lock below is verified
            # exactly as ever; the spawn is named only when ONE piece fits
            # the fragment — panel and top-edge hypotheses counted together,
            # since both are hypotheses about the same cells — and its cells
            # are never merged, only ``lock_cells`` are.
            read = _partial_piece(spawn_cells, stack_rows, unknown_rows)
            if not read.coherent:
                continue
            spawn_piece = read.piece
        elif spawn_piece.row >= SPAWN_ROWS:
            continue  # ghost-piece defense: spawns appear in the top rows
        lock_piece = _piece_at(lock_cells)
        if lock_piece is None:
            # A locked piece the covered region cut in half: its visible
            # cells are not a tetromino, but some piece completes them
            # under that region. Only those visible cells are merged below
            # — several pieces may fit, so the hidden ones stay a belief
            # rather than a guess dressed up as an observation.
            if not any(unknown_rows):
                continue
            if not _hidden_completions(lock_cells, unknown_rows, len(stack_rows)):
                continue
        elif last_falling is not None and lock_piece.piece != last_falling.piece:
            continue  # a piece cannot change identity between flight and lock
        if not _supported(lock_cells, stack_rows):
            continue  # a locked piece is at rest by definition
        valid.append((lock_cells, spawn_piece))
    if len(valid) == 2 and last_falling is not None:
        last_cells = set(last_falling.cells)
        overlapping = [entry for entry in valid if entry[0] & last_cells]
        if len(overlapping) == 1:
            valid = overlapping
    if len(valid) != 1:
        return None  # ambiguous: hold; the spawn descends and disambiguates
    lock_cells, spawn_piece = valid[0]
    merged = _rows_with(stack_rows, lock_cells)
    if any(row == FULL_ROW for row in merged):
        return None  # clear-flash frame: the full row will vanish
    return Explanation(FrameKind.LOCKED, merged, spawn_piece)


def _lock_with_clears(
    observed_rows: tuple[int, ...],
    stack_rows: tuple[int, ...],
    last_falling: FallingPiece | None,
    unknown_rows: tuple[int, ...],
) -> Explanation | None:
    """A lock that cleared lines (settled cells vanished).

    Tiered candidate placements, each verified exactly by :func:`_explains`.
    """
    # C1: the piece locked exactly where it was last observed (covers
    # lock-delay slides, tucks and T-spins whose final position was seen).
    # Skipped for a piece last seen half above the board region: its cells
    # are not all board cells (C2 below still drops it from that column).
    if last_falling is not None and last_falling.row >= 0:
        cells = last_falling.cells
        if all(not stack_rows[r] >> c & 1 for r, c in cells):
            merged = _rows_with(stack_rows, cells)
            if any(row == FULL_ROW for row in merged):
                s2 = clear_full_rows(merged)
                explained, residual = _explains(observed_rows, s2, unknown_rows)
                if explained:
                    return Explanation(FrameKind.LOCKED, s2, residual)
    board = Board(stack_rows)
    # C2: gravity drop from the last observed alignment.
    if last_falling is not None:
        rotation = ROTATIONS[last_falling.piece][last_falling.rotation_index]
        dropped = board.drop(rotation, last_falling.col)
        if dropped is not None and dropped.lines_cleared > 0:
            explained, residual = _explains(observed_rows, dropped.board.rows, unknown_rows)
            if explained:
                return Explanation(FrameKind.LOCKED, dropped.board.rows, residual)
    # C3: any gravity drop that clears lines, the last observed piece name
    # first — with no observation at all, all 7 names. This is how a line-
    # clearing lock by a never-observed piece still commits as verified
    # LOCKED rather than degrading to a board reset.
    names = list(PIECES)
    if last_falling is not None:
        names.remove(last_falling.piece)
        names.insert(0, last_falling.piece)
    for name in names:
        for rotation in ROTATIONS[name]:
            for col in range(WIDTH - rotation.width + 1):
                dropped = board.drop(rotation, col)
                if dropped is None or dropped.lines_cleared == 0:
                    continue
                explained, residual = _explains(observed_rows, dropped.board.rows, unknown_rows)
                if explained:
                    return Explanation(FrameKind.LOCKED, dropped.board.rows, residual)
    return None


def _descended_from(found: FallingPiece, top: int, left: int, width: int) -> bool:
    """Could ``found`` be the piece that vacated the box at ``(top, left)``?

    A piece leaves cells behind by MOVING, and between two ticks 67 ms
    apart it moves down and sideways, never UP — so the found piece must
    start at or below the vacated box's top row. That is the half that
    refuses the demonstrated false positives, all three of which vouch for
    a deletion with a piece somewhere above it.

    Sideways it must still be NEXT TO where it was: the two column spans
    touch or overlap. Measured over 364 consecutive same-piece
    observations across the four session replays, one tick moves a piece
    at most two columns and the two spans never part company — though
    they do merely touch (a vertical I stepping one column, an O stepping
    two), which is why "touch" and not "overlap" is the test. This game
    drags rather than drops, so a piece can cross its own width in a tick;
    it cannot cross the board.
    """
    if found.row < top:
        return False
    found_width = ROTATIONS[found.piece][found.rotation_index].width
    return found.col <= left + width and left <= found.col + found_width


def _carried_piece(
    observed_rows: tuple[int, ...],
    stack_rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
    max_missing_cells: int,
    entering_hint: str | None,
) -> Explanation | None:
    """The committed stack was holding a falling piece; take it back out.

    Self-healing, and the reason the absorbed-piece loop cannot persist
    however the piece got in. When settled cells vanish and no line clear
    explains it, one hypothesis is not "the world changed" but "the memory
    was wrong": the stack was carrying a piece that had not landed, and
    what looks like cells vanishing is that piece moving.

    The hypothesis is accepted only when it explains the WHOLE frame:

    - every vanished cell belongs to one tetromino-shaped set of cells the
      committed stack holds (so more than four missing cells — a line
      clear, rising garbage, a new board, a torn frame — is never this);
    - that piece is airborne in the stack without it (a piece resting on
      the stack is stack: it landed, and the memory is right);
    - and with it removed the frame reads as an ordinary FALLING frame
      whose piece has the SAME NAME *and could have got there from where
      the cells were*: at or below the rows they vacated, in columns that
      still touch theirs (:func:`_descended_from`).

    That last clause is the "it reappeared elsewhere" test, and the
    position half of it is what carries it. The name half alone vouches
    for nothing: any same-named tetromino ANYWHERE on the board answers
    for the deletion, so a real island at rows 8-9 of cols 0-1 was vouched
    for by an O at rows 0-1 of cols 8-9 — a piece that moved eight rows UP
    and eight columns across in one tick. Pieces fall; the memory being
    wrong means the piece moved DOWN out of cells the stack was holding,
    which is exactly what the absorbed-piece frames do (00309-00310 move
    straight down).

    The position test is also the only thing standing between this rule
    and a line clear's fade animation, which is the case ordering cannot
    reach: :func:`_lock_with_clears` explains the SETTLED post-clear
    frame, not a half-faded row, so a clearing row with its middle cells
    blanked arrives here as a <= 4-cell missing set that is airborne over
    the cavity beneath it and tetromino-shaped. Vouched for by the very
    piece whose lock caused the clear, it deleted the clearing row's cells
    and handed the solver a phantom gap to aim at; the piece is above the
    row it completed, so the position test refuses it.

    And it is what stops a subset from authorising four deletions: only
    the vanished cells need be part of the candidate (the real absorbed
    frames depend on that — a piece read in part is still the piece), so
    without it one dropped cell of a settled vertical I deletes all four
    while the other three are still lit on screen, read back as a piece
    that moved UP a row. Equality is not the fix; continuity is.

    Errs toward doing nothing: any ambiguity, any leftover, any resting
    candidate, and the frame stays unexplainable and the tracker holds.
    Unobservable cells are excluded from the vanished set upstream, so
    content that is merely covered is never evidence for this rule.
    """
    missing = _cells_from_rows(
        s & ~o & ~u for o, s, u in zip(observed_rows, stack_rows, unknown_rows, strict=True)
    )
    if not missing or len(missing) > MAX_IN_FLIGHT_CELLS:
        return None
    height = len(stack_rows)
    anchor_r, anchor_c = next(iter(missing))
    for rots in ROTATIONS.values():
        for rot in rots:
            for cell_r, cell_c in rot.cells:
                top, left = anchor_r - cell_r, anchor_c - cell_c
                if top < 0 or left < 0 or top + rot.height > height or left + rot.width > WIDTH:
                    continue
                cells = {(top + r, left + c) for r, c in rot.cells}
                if not missing <= cells:
                    continue
                if not all(_bit(stack_rows, cell) for cell in cells):
                    continue  # the stack never held this piece
                rest = _rows_without(stack_rows, cells)
                if _resting(cells, rest, unknown_rows):
                    continue  # it landed; the memory is right
                explanation = explain_grid(
                    observed_rows,
                    rest,
                    FallingPiece(rot.piece, rot.index, top, left),
                    max_missing_cells=max_missing_cells,
                    unknown_rows=unknown_rows,
                    entering_hint=entering_hint,
                    _heal=False,
                )
                if (
                    explanation.kind is FrameKind.FALLING
                    and explanation.falling is not None
                    and explanation.falling.piece == rot.piece
                    and _descended_from(explanation.falling, top, left, rot.width)
                ):
                    return Explanation(
                        FrameKind.FALLING,
                        rest,
                        explanation.falling,
                        hinted_name=explanation.hinted_name,
                        hint_refuted=explanation.hint_refuted,
                        entering_names=explanation.entering_names,
                    )
    return None


def explain_grid(
    observed_rows: tuple[int, ...],
    stack_rows: tuple[int, ...],
    last_falling: FallingPiece | None,
    *,
    max_missing_cells: int = 2,
    unknown_rows: tuple[int, ...] | None = None,
    entering_hint: str | None = None,
    _heal: bool = True,
) -> Explanation:
    """Explain one observed frame against the committed stack memory.

    ``observed_rows`` is the raw frame, ``stack_rows`` the committed stack,
    ``last_falling`` the last coherent raw falling observation (or ``None``).
    The falling piece is *derived* as ``observed & ~stack``; a frame no rule
    explains returns :attr:`FrameKind.UNEXPLAINED` and proposes nothing.

    ``unknown_rows`` marks cells the capture cannot observe (a game UI panel
    floating over the playfield). They are evidence for nothing: excluded
    from ``added`` and from the missing-cell count, free to stand in for a
    hypothesis' hidden cells, and never a reason to reset. With no
    unobservable cells (the default) every rule below is exactly the
    fully-observed one.

    ``entering_hint`` names the piece the NEXT preview says is entering the
    board. It is used for ONE thing: choosing between the tetrominoes that
    complete a fragment the top edge has cut in half, which no structural
    rule can separate (see :func:`_entering_piece`). It can never create,
    suppress or relocate an explanation.
    """
    unknown = unknown_rows if unknown_rows is not None else (0,) * len(observed_rows)
    added_rows = tuple(
        o & ~s & ~u for o, s, u in zip(observed_rows, stack_rows, unknown, strict=True)
    )
    n_miss = sum(
        (s & ~o & ~u).bit_count()
        for o, s, u in zip(observed_rows, stack_rows, unknown, strict=True)
    )
    added = _cells_from_rows(added_rows)
    hidden = any(unknown)

    # Step 1 — stack intact, or a small occlusion tolerated (trust memory
    # over vision for cursor/anti-aliasing dropouts; never on lock paths).
    if n_miss == 0 or (n_miss <= max_missing_cells and len(added) in (0, 4)):
        if not added:
            return Explanation(FrameKind.QUIET, stack_rows, None)
        piece = _piece_at(added)
        if piece is not None:
            # No floor-touch, connectivity, or support condition: a piece in
            # lock delay resting on the stack, sliding along it, leaning
            # against a column, or touching the bottom row is FALLING.
            return Explanation(FrameKind.FALLING, stack_rows, piece)

    # Step 1b — a piece the capture cannot see WHOLE shows 1-3 cells. That
    # is not a broken tetromino, it is a partly observed one: hidden under a
    # UI panel, or cut by the top edge of the board region on its way in. A
    # unique completion identifies it (and where it is), several mean the
    # frame is coherent but the piece unnameable — OCCLUDED holds state
    # rather than letting a stationary piece trip the reset debounce.
    if n_miss == 0 and 1 <= len(added) <= 3:
        # A panel hypothesis and a top-edge one are hypotheses about the
        # SAME cells, so they are counted together: asking the panel first
        # and answering from it alone reads a fragment three pieces could
        # be entering as a confident fourth (see :func:`_partial_piece`).
        read = _partial_piece(added, stack_rows, unknown, entering_hint)
        if read.coherent:
            kind = FrameKind.FALLING if read.piece is not None else FrameKind.OCCLUDED
            return Explanation(
                kind,
                stack_rows,
                read.piece,
                hinted_name=read.hinted,
                hint_refuted=read.refuted,
                entering_names=read.entering_names,
            )

    # Step 2 — lock revealed by the next spawn, no clears. A reveal whose
    # locked piece is partly hidden — or whose SPAWN is still cut by the top
    # edge — carries fewer than 4 + 4 added cells. (Fully observed, 5-7
    # added cells reach _lock_reveal and fail it, exactly as before: both
    # its rules need two whole tetrominoes.)
    partial = hidden or any(r == 0 for r, _ in added)
    if n_miss == 0 and (len(added) == 8 or (partial and 5 <= len(added) < 8)):
        locked = _lock_reveal(added, stack_rows, last_falling, unknown)
        if locked is not None:
            return locked

    # Step 3 — lock with line clears.
    if n_miss > 0:
        locked = _lock_with_clears(observed_rows, stack_rows, last_falling, unknown)
        if locked is not None:
            return locked

    # Step 3b — the committed stack was carrying a piece. Settled cells
    # cannot vanish, so a frame that says they did is either a new world or
    # a wrong memory; when one tetromino's worth of stack has gone missing
    # and that same piece is on the board somewhere else, the memory is
    # what was wrong. Correcting it here is what stops an absorbed piece
    # from resetting the board once a row, the whole way down. (Ordered
    # after the clear rules on purpose: a real clear reads as a clear.)
    if _heal and n_miss > 0:
        healed = _carried_piece(
            observed_rows, stack_rows, unknown, max_missing_cells, entering_hint
        )
        if healed is not None:
            return healed

    # Step 4 — no structural explanation. A first-class outcome, not an
    # error: piece entering from above, tearing, mid-fade clear frames,
    # rising garbage, piece+ghost pairs, unexplained missing cells.
    return Explanation(FrameKind.UNEXPLAINED, stack_rows, None)


# --- Next-piece preview ----------------------------------------------------

# A real preview crop is NOT a tight, centered, square box around the piece.
# The one in tests/fixtures/live_session is 94x94 and holds: a 4-cell piece
# of 19 px cells occupying about a fifth of the box, a faint grey "NEXT"
# label in the top-left corner, and white space. The label is the problem: it
# thresholds in at the SAME distance from the white ground as the piece does
# (measured on next_00043/00063: label 0.57, I-piece blue 0.75, O-piece
# yellow-green 0.53 on the score scale), so no threshold separates the two,
# and the mask's bounding box then runs from the label down to the piece:
# 45x79 for a piece that is 19x79. Measured on that box, every rotation
# either failed its aspect gate (a horizontal I: 0.44) or resampled to a
# shape that is no tetromino, and identify_next returned None on 96 of 96
# frames of the session — no lookahead at all, every hint 1-ply.
#
# So the piece is found by SHAPE, not by the crop: the mask's connected
# components are filtered down to the block-like ones, and the cell grid is
# derived from those blocks (their bands and their size) rather than from an
# even division of whatever the bounding box happens to span.

# A block (one cell, or several cells merged where a skin draws no gap)
# nearly fills its own bounding box: 1.0 for I and O, 4/6 for the merged
# T/S/Z/J/L bounding box, which is the floor any real block can reach.
# Glyph strokes, a box border and a gridline lattice all sit far below it
# (measured: the live "NEXT" fragments 0.60-0.75 but tiny, a hollow letter
# ~0.4), so this is what keeps a label out of the bounding box.
_MIN_BLOCK_SOLIDITY = 0.5

# ...and what keeps SMALL things out: a block is at worst a quarter of the
# largest block (one cell against four merged into one component), so
# anything under an eighth of it is not a block of the same rendering.
# Measured on the live crops: cells 357-361 px, label fragments 1-9 px.
_MAX_BLOCK_AREA_RATIO = 8.0

# Decisiveness of the per-cell sample (a central window of each derived
# cell, so insets and gridlines fall outside it). Measured over the whole
# synthetic style matrix and the live crops, an occupied cell samples 1.00
# and an empty one at most 0.40; the gap is what refuses text arranged in
# a row, which samples ~0.6 where a piece samples 1.0.
_MIN_CELL_FILL = 0.75
_MAX_EMPTY_FILL = 0.45

# Cells are square: the block size derived along x and the one along y must
# agree. Measured, they agree EXACTLY (ratio 1.000) on every synthetic style
# and every live crop; the slack is for antialiasing, not for shape. This is
# the test the old bounding-box aspect gate was approximating, except it is
# now measured on the blocks themselves, where an inset skin does not skew
# it — and it refuses tall solid blobs in a row (fake "letters" at 1.23).
_MAX_CELL_ASPECT = 1.25

# ...and cells are all the SAME size, so when the bands along an axis ARE the
# cells, they must agree with each other too — the same slack, for the same
# antialiasing reason. Taking the median instead (what this did before) hides
# exactly the shape that breaks the premise: one band holding something that
# is not one cell. A solid caption bar is that shape — it is block-like
# enough to survive :func:`_preview_blocks` (solid, and not small), and where
# it bridges the gap between two cells it merges them into one band three
# times the width of its neighbours, which the median reads as an ordinary
# cell and the aspect gate then waves through (measured: a 40x9 bar over a
# 4x1 I of 15 px cells is read as a confident J, on 120 of 495 bar
# geometries).
_MAX_BAND_SPREAD = 1.25

# The same premise about what separates them: the gap between two bands is
# the skin's inset or its gridline, so it is a FRACTION of a cell (measured
# over the style matrix and the live crops, at most 0.67 of one, and 0.05 on
# the live crops). A gap with room for a whole cell in it means the two bands
# are not adjacent cells of one piece — which is how a caption bar standing
# clear of the piece passes for a cell of its own.


def identify_next(image: NDArray[np.uint8], own_paint: OwnPaint | None = HINT_PAINT) -> str | None:
    """Recognize the piece shown in a next-piece preview image.

    Thresholds the image, keeps the block-like connected components (the
    piece; not a label, a border or a gridline lattice), derives the cell
    grid from those blocks, and returns the piece whose shape matches
    exactly — or ``None``. A piece drawn too pale for that threshold is
    read on a second pass, off the intermediate band (see
    :func:`_band_piece`), by the same shape rules.

    Game-agnostic by construction: nothing here assumes the piece is
    centered, that it fills the crop, that the crop is square, or that the
    crop holds nothing else. What it assumes is that a tetromino is drawn
    as square cells of one size, which is the same premise the board
    reader works from.

    ``None`` is a first-class answer. A confidently WRONG piece is worse:
    it feeds a 2-ply hint that plans around a piece the game never deals,
    and it NAMES the half-visible piece entering the board (see
    :func:`_partial_piece`), so one wrong reading of the box becomes a
    wrong hint about a piece that is on screen.

    Scoring is per-pixel distance from the box's own background color,
    estimated as the per-channel median of ALL pixels: a preview box is
    majority-background (a piece is at most 4 cells of a >= 24-cell box),
    and an arbitrary crop has no meaningful top row to sample instead.

    ``own_paint`` is the coach's own hint fill, for the one thing in the
    box that must never be read as a piece however block-like it is: this
    tool's own overlay, which lands on the preview panel whenever the
    hint it draws is in the board corner the panel floats over.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    background = np.median(img.reshape(-1, img.shape[2]), axis=0)
    scores = _distance_scores(img, background)
    flat = scores.ravel()
    if float(flat.max()) - float(flat.min()) >= MIN_SPREAD:
        # Histogram Otsu: the input is every pixel of the preview image,
        # far too many for the exact small-N variant's per-sample loop.
        # The mask is additionally floored at MIN_SPREAD: a pixel closer
        # to the background than the uniformity floor is background by
        # the pipeline's own definition. Without the floor, a dense
        # gridline lattice (a mid class between background and piece,
        # lifted by the sqrt compression) can tip pixel-scale Otsu into
        # splitting background|(gridlines+piece) and ruin the bounding
        # box; every SOLID piece color sits well above the floor (see the
        # measured anchors on MIN_SPREAD).
        solid = scores > max(otsu_threshold_hist(flat), MIN_SPREAD)
        piece = _piece_from_mask(solid)
        if piece is not None:
            # ...and our own paint is refused HERE too, not only in the
            # band. The hint fill lands in the band over a LIGHT box
            # (0.321 over white), which is what made the band the place
            # to ask — but the composite is a distance from the box's
            # ground, not a constant: over a BLACK box the same fill
            # scores 0.374, above MIN_SPREAD entirely, so it arrives as
            # a solid class and this branch names it. Measured over
            # black and near-black box grounds it scores 0.357-0.374 and
            # an O-, T- or I-shaped hint was read as 'O', 'T', 'I': the
            # exact name of the placement the coach is pointing at,
            # reported as the piece coming next. A dark theme is not an
            # edge case, and the rule belongs to both readings of the
            # box rather than to the pass that happened to need it
            # first.
            if _is_own_paint(img, background, solid, own_paint):
                return None
            return piece
    # ...and a piece that does NOT sit above the floor is the whole of
    # what is left to read. Below the floor there is no spread to gate on
    # either: a pale piece on a box with no caption in it moves the box's
    # extremes not at all (measured on the live crops: 0.346 against a
    # 0.35 floor), which is why the band is asked even where the old
    # early return called the box empty.
    return _band_piece(img, scores, background, own_paint)


def _band_piece(
    img: NDArray[np.uint8],
    scores: NDArray[np.float32],
    background: NDArray[np.float64],
    own_paint: OwnPaint | None,
) -> str | None:
    """The piece drawn in the INTERMEDIATE band, or ``None``.

    The band is ``[_GHOST_SEPARATION, MIN_SPREAD)`` — clear of the
    background, below the color distance that makes a solid piece — and
    it is where this game draws a pale periwinkle piece: measured on the
    committed crops, 0.295-0.330 against a background cluster at
    0.00-0.05, where a solid piece reads 0.53-0.76. ``vision.grid``
    already refuses to let a threshold decide that band on the BOARD
    (see :func:`~tetris_coach.vision.grid._classify_scored`); this is the
    same refusal for the box. Measured before it, the box went blind for
    as long as a pale piece sat in it: 226 of 705 frames of a real
    session read as an empty box, in runs up to 45 frames.

    The band is not simply handed to the shape rules, because a preview
    crop is not a board: it carries furniture that scores in the band
    too, which a board cell's patch mean never sees. On these crops the
    box's own hairline border and its gridlines sit at 0.180-0.190 —
    under the piece, inside the band, and touching it. Taken whole, the
    band merges the piece's cells into one component through that border
    and there are no blocks left to read (measured: 0 of 76 pale crops
    readable). So the band is SPLIT, by the same method the frame is:
    Otsu over the band's own pixels, and the upper class — the level
    nearest a real piece color — is the candidate. Measured, that split
    lands at 0.219-0.291 on every committed crop, between the furniture
    and the piece every time.

    What it CANNOT contain is the solid class, and that is what keeps the
    caption out rather than a rule about captions: a "NEXT" caption's
    core scores 0.586, above the floor entirely, so only its antialiased
    skirt reaches the band — a hollow outline the block filter drops.
    Admitted instead as a lowered threshold, the caption composes into
    one solid blob that survives :func:`_preview_blocks` and drags the
    bounding box off the piece (measured on next_00580: block-like at
    area 79 against the piece's 357, three row bands where a T has two,
    and no match at all).

    A band with nothing to split — one flat level, no furniture in it and
    no antialiasing around it — is taken whole: there is no upper class
    when there is only one class, and a box that holds nothing but the
    piece is the easy case, not a refusal.

    Which way it errs: toward ``None``. A band this pass cannot resolve
    into exactly one piece — furniture mixed into the level, a piece half
    drawn, a box mid-wipe — gets the same silence the threshold gave it,
    and the one thing it refuses outright is our own paint. It never
    overrides a piece the threshold could read: it is only ever asked
    after that reading came back empty-handed, so every frame that was
    readable before is byte-identical, answer for answer.
    """
    band = (scores >= _GHOST_SEPARATION) & (scores < MIN_SPREAD)
    if not bool(band.any()):
        return None
    candidate = band & (scores > otsu_threshold_hist(scores[band]))
    if not bool(candidate.any()):
        candidate = band
    if _is_own_paint(img, background, candidate, own_paint):
        return None
    return _piece_from_mask(candidate)


def _is_own_paint(
    img: NDArray[np.uint8],
    background: NDArray[np.float64],
    candidate: NDArray[np.bool_],
    paint: OwnPaint | None,
) -> bool:
    """Is the band's candidate level this tool's own hint fill?

    The coach draws its hint over the game and captures the screen again,
    so its own overlay comes back round as input — and the preview panel
    floats over the top corner of the playfield, so a hint drawn in that
    corner is drawn over the BOX. It is a tetromino, drawn as square
    cells, in the one place a tetromino is expected: nothing in the shape
    rules can tell it from the piece the game dealt, and believing it
    would name the piece the coach is pointing at as the piece coming
    next.

    Its color, unlike the game's, is known in advance
    (:class:`~tetris_coach.vision.grid.OwnPaint`), and over a light box
    it composites to exactly the band this pass reads: the shipped cyan
    at 0.18 over white scores 0.321, four thousandths from the pale piece
    at 0.325. Distance from the composite separates them where the score
    cannot — measured over every committed preview crop, the nearest real
    band pixel to the composite stands 2.6x the tolerance clear of it
    (the tolerance being 8.0 uint8 units, and the same one the board's
    own paint rule is measured against).

    A match REFUSES the band rather than deleting it, which is the answer
    ``vision.grid`` gives its own unnameable paint: a box holding our
    hint is a box whose real contents we cannot see, and ``None`` is
    already this module's word for that. Majority rather than any pixel,
    so a hint overlapping the box's edge cannot suppress a real piece
    beside it.
    """
    if paint is None or len(paint.color) != int(img.shape[-1]):
        return False
    color = np.asarray(paint.color, dtype=np.float64)
    pixels = np.asarray(img, dtype=np.float64)
    for ground in _grounds(pixels, background, candidate):
        composite = ground + paint.opacity * (color - ground)
        diff = pixels - composite
        painted = np.sqrt(np.sum(diff * diff, axis=-1)) <= paint.tolerance
        if int((painted & candidate).sum()) * 2 >= int(candidate.sum()):
            return True
    return False


def _grounds(
    pixels: NDArray[np.float64],
    background: NDArray[np.float64],
    candidate: NDArray[np.bool_],
) -> list[NDArray[np.float64]]:
    """The box's background levels, for the composite to be computed over.

    The arithmetic in :func:`_is_own_paint` needs the color the paint
    landed ON, and :func:`identify_next` estimates one: the per-channel
    median of ALL the crop's pixels. That estimate is the right one for
    the job it was written for (scoring distance from the background) and
    it is exactly one color, which is one too few here — a preview box
    with a panel and an inner WELL of a second shade is the ordinary
    look, and the median then sits on whichever of the two has more
    pixels while the hint is composited over the other. Measured with the
    hint drawn over the well: the two composites stand ~21 uint8 units
    apart against a tolerance of 8.0, so the rule said "not our paint"
    and the box was named as the piece the coach is pointing at —
    'O'/'T'/'I' on a white panel around a light well, on a near-black
    panel around a darker well, and on a mid-grey panel around a darker
    well alike. The same box in ONE shade was refused correctly, which
    is the whole difference.

    So the grounds are the LEVELS of the crop, not its average: the
    median, plus the median of each Otsu class of everything that is not
    the candidate itself. A one-shade box gives three colors that are the
    same color and the reading does not move; a two-shade box gives the
    panel and the well, and the paint is recognized over either.

    Widening what the rule may match is safe in the direction that
    matters here because the rule REFUSES rather than names: its cost is
    a box read as ``None``, never a wrong piece. It is still bounded by
    measurement — over every committed preview crop (1115 candidates,
    both passes), the nearest pixel of a real candidate to the composite
    of ANY of these grounds stands 19.1 units off it, 2.4x the tolerance
    and the same order as the 2.6x the single-ground rule was measured
    at; not one candidate has a single pixel within tolerance, against
    the majority the rule asks for.
    """
    grounds = [np.asarray(background, dtype=np.float64)]
    rest = pixels[~candidate]
    if rest.size:
        scores = _distance_scores(rest, grounds[0]).ravel()
        level = otsu_threshold_hist(scores)
        for side in (scores <= level, scores > level):
            if bool(side.any()):
                grounds.append(np.asarray(np.median(rest[side], axis=0), dtype=np.float64))
    return grounds


def _piece_from_mask(mask: NDArray[np.bool_]) -> str | None:
    """The one piece whose shape the foreground ``mask`` draws, or ``None``.

    The half of :func:`identify_next` that works on shape alone: keep the
    block-like components, derive the cell grid from them, and accept a
    rotation only when every one of its cells is filled and every cell
    outside it is not. Split out from the thresholding half so that the
    band pass (:func:`_band_piece`) is read by exactly these rules.
    """
    blocks = _preview_blocks(mask)
    if blocks is None:
        return None
    ys, xs = np.nonzero(blocks)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = blocks[y0:y1, x0:x1]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None

    rows_profile = crop.any(axis=1)
    cols_profile = crop.any(axis=0)
    matched: set[str] = set()
    for rots in ROTATIONS.values():
        for rot in rots:
            down = _axis_grid(rows_profile, rot.height)
            across = _axis_grid(cols_profile, rot.width)
            if down is None or across is None:
                continue
            aspect = across.cell / down.cell
            if not 1.0 / _MAX_CELL_ASPECT <= aspect <= _MAX_CELL_ASPECT:
                continue
            means = _cell_means(crop, down, across)
            expected = np.zeros((rot.height, rot.width), dtype=bool)
            for r, c in rot.cells:
                expected[r, c] = True
            if not np.array_equal(means > 0.5, expected):
                continue
            if float(means[expected].min()) < _MIN_CELL_FILL:
                continue
            empty = means[~expected]
            if empty.size and float(empty.max()) > _MAX_EMPTY_FILL:
                continue
            matched.add(rot.piece)
    # Two different pieces reading the same blocks is an ambiguity no
    # shape rule can settle (it cannot happen between two rotations of one
    # grid, whose cell patterns differ by construction), so say nothing.
    if len(matched) != 1:
        return None
    return next(iter(matched))


def _preview_blocks(mask: NDArray[np.bool_]) -> NDArray[np.bool_] | None:
    """The block-like part of a preview mask: the piece, without the furniture.

    A preview box holds more than the piece — a "NEXT" label, a border, a
    gridline lattice — and those score as far from the background as the
    piece does, so they survive the threshold and, left in, drag the
    bounding box off the piece entirely. They are dropped on shape:

    - a block nearly fills its own bounding box (:data:`_MIN_BLOCK_SOLIDITY`),
      which a glyph stroke, a hollow border and a lattice never do;
    - a block is within :data:`_MAX_BLOCK_AREA_RATIO` of the largest one,
      which small furniture next to a big piece never is.

    Returns ``None`` when nothing block-like is left (an empty preview box
    holding only its label).
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    solid: list[tuple[int, int]] = []
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= _MIN_BLOCK_SOLIDITY * width * height:
            solid.append((area, index))
    if not solid:
        return None
    largest = max(area for area, _ in solid)
    kept = [index for area, index in solid if area * _MAX_BLOCK_AREA_RATIO >= largest]
    return np.isin(labels, kept)


@dataclass(frozen=True)
class _AxisGrid:
    """Where one axis' cells sit, derived from the blocks along it."""

    centers: tuple[float, ...]  # cell centers, in crop pixels
    half: float  # half-width of the sample window around a center
    cell: float  # block size along this axis, for the squareness test


def _bands(profile: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Maximal runs of rows/columns that hold any foreground pixel."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, filled in enumerate(profile):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(profile)))
    return runs


def _axis_grid(profile: NDArray[np.bool_], count: int) -> _AxisGrid | None:
    """The ``count`` cell positions along one axis, derived from the blocks.

    EVERY row and column of a tetromino's bounding box holds at least one
    of its cells (that is what a bounding box is), so the blocks say how
    many cells the axis has:

    - ``count`` bands: a skin that leaves a gap between cells. The cells
      are the bands — centers and size read straight off them, which is
      what makes an inset skin readable (dividing the bounding box evenly
      instead puts the sample window off the cell by half the inset, and
      that is what made small inset previews unreadable).
    - one band: a skin that draws cells flush, where a band cannot be a
      cell and an even division of the extent is exact (cell size = pitch).
    - anything else: the blocks contradict this hypothesis — a broken or
      missing cell — and there is nothing to say. Refused, not guessed.
    """
    bands = _bands(profile)
    if len(bands) == count:
        sizes = [stop - start for start, stop in bands]
        if min(sizes) * _MAX_BAND_SPREAD < max(sizes):
            return None  # a band that is not one cell: not this hypothesis
        gaps = [bands[index + 1][0] - bands[index][1] for index in range(len(bands) - 1)]
        if gaps and max(gaps) >= min(sizes):
            return None  # a gap wide enough to hold a cell is not a skin's gap
        centers = tuple((start + stop - 1) / 2.0 for start, stop in bands)
        size = float(np.median(sizes))
        return _AxisGrid(centers, max(size * 0.4, 0.5), size)
    if len(bands) == 1:
        pitch = len(profile) / count
        centers = tuple((index + 0.5) * pitch for index in range(count))
        return _AxisGrid(centers, max(pitch * 0.25, 0.5), pitch)
    return None


def _cell_means(crop: NDArray[np.bool_], down: _AxisGrid, across: _AxisGrid) -> NDArray[np.float64]:
    """Foreground fraction of a central window of each derived cell.

    A central window rather than the whole cell: the gap an inset skin
    leaves between cells, and a gridline drawn over one, are at the cell's
    EDGE, so sampling the middle reads occupancy without reading the skin.
    """
    height, width = crop.shape
    means = np.empty((len(down.centers), len(across.centers)), dtype=np.float64)
    for r, center_y in enumerate(down.centers):
        top = max(0, round(center_y - down.half))
        bottom = min(height, max(top + 1, round(center_y + down.half) + 1))
        for c, center_x in enumerate(across.centers):
            left = max(0, round(center_x - across.half))
            right = min(width, max(left + 1, round(center_x + across.half) + 1))
            means[r, c] = crop[top:bottom, left:right].mean()
    return means

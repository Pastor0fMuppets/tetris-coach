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

import numpy as np
from numpy.typing import NDArray

from ..core.board import FULL_ROW, WIDTH, Board
from ..core.pieces import PIECES, ROTATIONS
from .grid import MIN_SPREAD, _distance_scores, otsu_threshold_hist

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


def _entering_piece(
    fragment: set[Cell],
    stack_rows: tuple[int, ...],
    unknown_rows: tuple[int, ...],
) -> tuple[bool, FallingPiece | None]:
    """``(the fragment is a piece entering from above, its name when unique)``.

    Ambiguity is reported as ``(True, None)``: the frame is coherent — some
    piece is entering — but two horizontally adjacent cells fit an O, an S,
    a Z, a J, an L and a T alike, and a guessed name means a guessed hint.
    Holding beats guessing; the piece names itself one row later.
    """
    completions = _clipped_completions(fragment, stack_rows, unknown_rows)
    if not completions:
        return False, None
    if len(completions) == 1:
        return True, next(iter(completions))
    return True, None


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
    entering, entering_piece = _entering_piece(residual, s2_rows, unknown_rows)
    if entering:
        return True, entering_piece
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
            # The spawn revealing the lock may itself be cut by the top edge
            # of the board region, showing 1-3 cells (this game spawns a
            # piece the moment the previous one is dropped, and the fragment
            # SITS there). The lock below is verified exactly as ever; the
            # entering piece is named only when one tetromino fits it, and
            # its cells are never merged — only ``lock_cells`` are.
            entering, spawn_piece = _entering_piece(spawn_cells, stack_rows, unknown_rows)
            if not entering:
                continue
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


def explain_grid(
    observed_rows: tuple[int, ...],
    stack_rows: tuple[int, ...],
    last_falling: FallingPiece | None,
    *,
    max_missing_cells: int = 2,
    unknown_rows: tuple[int, ...] | None = None,
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
        if hidden:
            completions = _hidden_completions(added, unknown, len(observed_rows))
            if len(completions) == 1:
                piece = _piece_at(next(iter(completions)))
                if piece is not None:
                    return Explanation(FrameKind.FALLING, stack_rows, piece)
            if completions:
                return Explanation(FrameKind.OCCLUDED, stack_rows, None)
        # The panel is asked first and answers conclusively, so a fragment
        # it can explain is explained there and nothing a panel selection
        # used to do changes. Only what the panel cannot explain — every
        # fragment in a session with no panel at all — reaches the top edge.
        entering, piece = _entering_piece(added, stack_rows, unknown)
        if entering:
            kind = FrameKind.FALLING if piece is not None else FrameKind.OCCLUDED
            return Explanation(kind, stack_rows, piece)

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

    # Step 4 — no structural explanation. A first-class outcome, not an
    # error: piece entering from above, tearing, mid-fade clear frames,
    # rising garbage, piece+ghost pairs, unexplained missing cells.
    return Explanation(FrameKind.UNEXPLAINED, stack_rows, None)


def identify_next(image: NDArray[np.uint8]) -> str | None:
    """Recognize the piece shown in a next-piece preview image.

    Thresholds the image, crops the foreground to its bounding box,
    resamples it to each candidate rotation's cell grid, and returns the
    piece whose shape signature matches exactly (or ``None``).

    Scoring is per-pixel distance from the box's own background color,
    estimated as the per-channel median of ALL pixels: a preview box is
    majority-background (a piece is at most 4 cells of a >= 24-cell box),
    and an arbitrary crop has no meaningful top row to sample instead.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    background = np.median(img.reshape(-1, img.shape[2]), axis=0)
    scores = _distance_scores(img, background)
    flat = scores.ravel()
    if float(flat.max()) - float(flat.min()) < MIN_SPREAD:
        return None  # empty preview box
    # Histogram Otsu: the input is every pixel of the preview image, far
    # too many for the exact small-N variant's per-sample loop. The mask
    # is additionally floored at MIN_SPREAD: a pixel closer to the
    # background than the uniformity floor is background by the pipeline's
    # own definition. Without the floor, a dense gridline lattice (a mid
    # class between background and piece, lifted by the sqrt compression)
    # can tip pixel-scale Otsu into splitting background|(gridlines+piece)
    # and ruin the bounding box; every real piece color sits well above
    # the floor (see the measured anchors on MIN_SPREAD).
    mask = scores > max(otsu_threshold_hist(flat), MIN_SPREAD)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = mask[y0:y1, x0:x1]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None

    best: tuple[float, str] | None = None
    for rots in ROTATIONS.values():
        for rot in rots:
            resampled = _resample_to_cells(crop, rot.height, rot.width)
            if resampled is None:
                continue
            occupancy, fit = resampled
            expected = np.zeros((rot.height, rot.width), dtype=bool)
            for r, c in rot.cells:
                expected[r, c] = True
            if not np.array_equal(occupancy, expected):
                continue
            if best is None or fit > best[0]:
                best = (fit, rot.piece)
    return best[1] if best else None


def _resample_to_cells(
    crop: NDArray[np.bool_], rows: int, cols: int
) -> tuple[NDArray[np.bool_], float] | None:
    """Block-average ``crop`` onto a (rows, cols) cell grid.

    Returns ``(occupancy, fit)`` where ``fit`` measures how decisive the
    block means are (1.0 = every block fully on or off), or ``None`` when
    the crop's aspect ratio is far from the target grid's.
    """
    h, w = crop.shape
    aspect = (w / h) / (cols / rows)
    if not 0.6 <= aspect <= 1.7:
        return None
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    means = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = crop[ys[r] : max(ys[r] + 1, ys[r + 1]), xs[c] : max(xs[c] + 1, xs[c + 1])]
            means[r, c] = block.mean()
    occupancy = means > 0.5
    fit = float(np.abs(means - 0.5).mean() * 2.0)
    return occupancy, fit

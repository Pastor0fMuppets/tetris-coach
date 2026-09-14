"""Occupancy classification of a board-region image.

Game-agnostic: works on *occupancy*, not colors — and not on any absolute
brightness scale either. Each cell of the (rows x cols) grid is sampled at
the central portion of its patch (skipping gridlines) and scored by its
color's Euclidean distance from a per-frame estimate of the board's own
background color, so 0 means "at the background" BY CONSTRUCTION whatever
the theme: dark boards with bright pieces, white boards with colored
pieces, and colored backgrounds all score the same way. The darker-
background restriction of earlier versions is lifted at this layer.

The background estimate comes from one of two places: a caller-supplied
``background`` color (cross-frame memory), or, absent one, the
per-channel median of the TOP ROW's cell colors (see
:func:`cell_scores` for the gravity-prior argument and its limits). The
rows x cols cell scores are then split by Otsu's method into empty/occupied
classes; distance-from-background makes the polarity fixed by
construction (high score = occupied) — provided the estimate really is
the background, which is exactly what the top-row prior cannot guarantee
when a stack legally reaches the visible top row (see
:func:`classify_grid`'s top-row cap).

``unobservable_cells`` names board cells the capture can never read
because a game UI panel floats over them (ROAS Stacker's NEXT preview
sits on the board's top corner; ``app.compute_overlap_mask`` computes
the set). Their pixels are the panel's, not the board's, so they are
excluded from the top-row background sample and from the top-row cap —
otherwise a permanently covered top-row cell is a permanently "occupied"
top row, the cap fires on every frame, and :class:`GridClassifier`'s
memory (which only anchors from ACCEPTED frames) can never form. The
default, an empty set, leaves every behavior exactly as it was.

Channel order does not matter (BGR vs RGB): Euclidean distance and the
per-channel median are permutation-equivariant in the channel axis.
A 2-D grayscale image is simply the single-channel (C=1) case.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT

# Uniformity floor on the sqrt-compressed distance-score scale: below this
# spread there are no two classes for Otsu to separate (see classify_grid).
# 0.35 on this scale is a raw normalized distance of ~0.12, i.e. ~54 uint8
# Euclidean units. Measured anchors:
#   - empty-board cell scores <= 0.06 (patch means average the noise out);
#   - blank_light_preview.png fixture, per-pixel max: 0.26;
#   - classic-dark gridline pixels: 0.36;
#   - faintest piece color of the synthetic light-style battery: 0.47.
# Chosen low (0.35 over 0.40) because the floor's failure asymmetry favors
# low: a theme whose pieces score just above it is still read exactly,
# while a floor above the piece scores would declare the board empty at a
# gate-passing confidence. Shared with pieces_vision.identify_next so the
# two uniform-region rules cannot drift apart. Doubles as the absolute
# ceiling for the EMPTY class in _classify_scored's re-split: a cell
# scoring below it is at the background by construction, one above it is
# not — so an Otsu split that leaves super-floor cells in the empty class
# split piece-vs-piece and is re-anchored at the floor.
MIN_SPREAD = 0.35

# Confidence reported for a uniform region at the background color — an
# empty board, whatever color the theme paints it. There is no occupied/
# empty split to measure, but the reading is structurally grounded (every
# cell sits at the frame's own background estimate), so report enough
# confidence to pass any sane frame gate (well above
# CoachConfig.min_confidence) while staying below a cleanly separated
# two-class frame: a board wipe must reach the tracker whatever color the
# empty board is.
_UNIFORM_EMPTY_CONFIDENCE = 0.5

# Cells in a tetromino — also the most rows one can span. The size of the
# ONLY thing that may legitimately contaminate the top row without breaking
# the background estimate: a piece in flight. See :func:`_top_row_vouchable`.
_PIECE_CELLS = 4

# How many consecutive impossible frames a background memory survives
# without a second opinion against it. A reading that claims a completed
# row describes a board the game cannot show
# (:func:`_claims_a_completed_row`); a run of them is a line clear or a
# game-over fill caught mid-animation, an endless stream is the anchor
# itself, inverted. This is only the SAFETY NET — an inverted anchor is
# normally caught on the first ordinary frame, by the second opinion
# GridClassifier._impossible_frame takes — so it is set long rather than
# short: at CoachConfig.poll_rate (15 fps) this is ~1 s, comfortably past
# any clear animation, and it costs a wedged session that long only when
# no frame in it can be read from scratch at all.
_IMPOSSIBLE_FRAME_LIMIT = 15


def _cell_colors(
    image: NDArray[np.uint8],
    rows: int,
    cols: int,
    margin: float,
) -> NDArray[np.float32]:
    """Mean color of the central patch of every cell: (rows, cols, C) float32.

    ``margin`` is the fraction of each cell inset on every side before
    sampling, which skips gridlines and cell borders. A 2-D grayscale
    image is treated as (H, W, 1).
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w = int(img.shape[0]), int(img.shape[1])
    if h < rows or w < cols:
        raise ValueError(f"image {w}x{h} too small for a {cols}x{rows} grid")
    ys = np.linspace(0, h, rows + 1)
    xs = np.linspace(0, w, cols + 1)
    out = np.empty((rows, cols, int(img.shape[2])), dtype=np.float32)
    for r in range(rows):
        cell_h = float(ys[r + 1] - ys[r])
        y0 = round(float(ys[r]) + cell_h * margin)
        y1 = max(y0 + 1, round(float(ys[r + 1]) - cell_h * margin))
        for c in range(cols):
            cell_w = float(xs[c + 1] - xs[c])
            x0 = round(float(xs[c]) + cell_w * margin)
            x1 = max(x0 + 1, round(float(xs[c + 1]) - cell_w * margin))
            out[r, c] = img[y0:y1, x0:x1].mean(axis=(0, 1))
    return out


def _distance_scores(
    colors: NDArray[np.uint8] | NDArray[np.float32],
    background: NDArray[np.float64] | NDArray[np.float32],
) -> NDArray[np.float32]:
    """sqrt-compressed normalized Euclidean distance from ``background``.

    ``colors`` has channels on the last axis (any leading shape: cell-color
    arrays and per-pixel arrays alike); the raw distance is normalized by
    its maximum possible value ``255 * sqrt(C)`` and square-rooted, giving
    a score in [0, 1] with no clipping needed.

    The sqrt compression is load-bearing, not cosmetic: it expands the
    gap between the background cluster (near 0) and the NEAREST piece
    color while compressing the spread among distant piece colors, so a
    palette containing one near-background color and one far color still
    splits empty-vs-piece rather than piece-vs-piece under Otsu.
    """
    arr = np.asarray(colors, dtype=np.float32)
    diff = arr - np.asarray(background, dtype=np.float32)
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    max_dist = 255.0 * float(np.sqrt(arr.shape[-1]))
    return np.sqrt(dist / max_dist).astype(np.float32)


def _top_row_background(
    colors: NDArray[np.float32],
    unobservable: frozenset[tuple[int, int]],
) -> NDArray[np.float64]:
    """Per-channel median of the OBSERVABLE top-row cell colors.

    A cell the capture cannot read holds the covering panel's pixels, so
    including it biases the median toward a color that is not on the board
    at all — and a panel is opaque and permanent, so the bias is on every
    frame of the session. Falls back to the whole top row if every one of
    its cells is declared unobservable: nothing better is available, and
    the value is only ever used to keep the arithmetic total — a board
    whose entire top row is covered has no top-row sample at all, and
    :func:`_top_row_vouchable` refuses every such frame outright.
    """
    top = colors[0]
    if unobservable:
        keep = [c for c in range(int(top.shape[0])) if (0, c) not in unobservable]
        if keep:
            top = top[keep]
    return np.asarray(np.median(top, axis=0), dtype=np.float64)


def _flood(solid: NDArray[np.bool_], seeds: list[tuple[int, int]]) -> NDArray[np.bool_]:
    """Cells of ``solid`` reachable from ``seeds`` by 4-connected steps."""
    rows, cols = int(solid.shape[0]), int(solid.shape[1])
    seen = np.zeros((rows, cols), dtype=np.bool_)
    stack = [(r, c) for r, c in seeds if solid[r, c]]
    for cell in stack:
        seen[cell] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and solid[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _support(
    occupancy: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split the occupied cells into ``(airborne, grounded)``.

    Grounded means 4-connected to the bottom row through occupied cells:
    a stack rests on the floor, and a piece that lands joins it (that is
    what landing IS). Connectivity, not the cell directly below, is what
    decides — an overhang has air under it and is still part of the stack
    it juts out of, so support cannot be read column by column.

    Cells the capture cannot read are evidence for neither side: they
    neither ground a component nor conduct support to one, exactly as
    they are left out of every other rule here.
    """
    rows, cols = int(occupancy.shape[0]), int(occupancy.shape[1])
    solid = np.array(occupancy, dtype=np.bool_)
    for r, c in unobservable:
        if 0 <= r < rows and 0 <= c < cols:
            solid[r, c] = False
    grounded = _flood(solid, [(rows - 1, c) for c in range(cols)])
    return solid & ~grounded, grounded


def _over_the_void(
    airborne: NDArray[np.bool_],
    grounded: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Airborne cells with no grounded cell anywhere below them.

    The strongest form of "hanging": not merely unsupported, but with
    nothing under it all the way down to the floor. A stack band cut
    loose by a buried hole is airborne and NOT over the void — the rest
    of its own column still stands under it — which is why this, and not
    the raw airborne count, is what a board-wide budget can be spent on.
    """
    supported_below = np.zeros_like(grounded)
    # reach[r, c] = "a grounded cell lies at or below row r in column c"
    reach = np.logical_or.accumulate(grounded[::-1], axis=0)[::-1]
    supported_below[:-1] = reach[1:]
    return airborne & ~supported_below


def _claims_a_completed_row(
    occupancy: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
) -> bool:
    """Does this reading describe a board the game cannot be showing?

    A completed row clears the instant it completes, so no frame of a
    running game shows one. That makes "a row is occupied end to end" a
    CONTRADICTION rather than a prior — the only rule here that reads the
    board without assuming anything about where the background is, which
    is exactly what is needed to catch a background estimate that has
    gone inverted.

    It is the ordinary frames that give an inverted anchor away, and they
    give it away loudly: the air above a normal stack is every cell of
    every row above it, so inverted, each of those rows reads as a
    completed line. (Measured on the wedge this exists for: a session
    anchored on a piece color read 7 of 12 rows completed on every
    ordinary play frame.) The near-top-out frames that CAUSE the
    inversion show no completed row either way, which is why nothing can
    be concluded from them and why this must be checked on the frames
    that follow.

    Only FULLY observable rows count. Where a UI panel covers part of a
    row the cells behind it are unknown, and versus-mode garbage is 9/10
    filled — a garbage row whose one gap happens to sit behind the panel
    is a legal row that merely LOOKS complete, so rows the panel touches
    are not evidence of anything.
    """
    rows, cols = int(occupancy.shape[0]), int(occupancy.shape[1])
    covered = np.zeros((rows, cols), dtype=np.bool_)
    for r, c in unobservable:
        if 0 <= r < rows and 0 <= c < cols:
            covered[r, c] = True
    return bool((occupancy.all(axis=1) & ~covered.any(axis=1)).any())


def _top_row_vouchable(
    occupancy: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
) -> bool:
    """May a self-estimated reading be vouched for, given its own top row?

    The top-row median is the background as long as a MAJORITY of the
    sampled cells really are background; past that the estimate locks onto
    piece colors and every score in the frame inverts (see
    :func:`cell_scores`). The reading itself is the only evidence
    available, so it is asked to show what a board carrying ONE PIECE IN
    FLIGHT would show, and nothing else:

    0. The top row must have been SAMPLED at all. With every one of its
       cells declared unobservable there is no sample:
       :func:`_top_row_background` has nothing to take a median of and
       falls back to the covering panel's own pixels, so the "background"
       is a color the board never shows and the split can land anywhere.
       This is not a corner case of a corner case — it is what
       ``compute_overlap_mask`` returns for any game whose NEXT queue is a
       horizontal bar across the top of the playfield, e.g.
       ``Rect(228, 314, 480, 576)`` with ``Rect(220, 310, 500, 50)`` at
       rows=12, which masks (0,0)..(0,9). Measured on such a frame: a
       12x10 light-theme board filled from row 8 down, its top row painted
       a UI-panel color, read as rows 1-11 FULLY OCCUPIED — 74 observable
       cells wrong — at confidence 0.72, five times the frame gate.
    1. A strict majority of the OBSERVABLE top-row cells must read empty.
       More cells occupied than not contradicts the estimator's own
       premise outright — and no single falling piece can put that many
       cells in one row.
    2. The occupied top-row cells must BE that piece: every one of them
       airborne (:func:`_support`), all in the same airborne component,
       and that component no bigger than a tetromino. A stack reaching
       row 0 is grounded instead — and so, transitively, is an inverted
       reading's "piece", because the cells it calls occupied are the
       true background, which reaches the floor down some column often
       enough.
    3. Board-wide, no more than a tetromino's worth of cells may hang
       OVER THE VOID (:func:`_over_the_void`) — airborne with nothing
       below them at all. One piece falling over empty columns is the
       only thing a legal board can show there.

    Rule 2 is what separates the two states a count alone cannot: a piece
    spawning or falling through row 0 hangs over the board, a stack
    reaching row 0 stands on it. (A count cannot separate them at all:
    with m of N top-row cells truly non-background, a benign reading
    shows m occupied and an inverted one shows N - m, and inversion needs
    m > N/2, so both land under N/2. Measured: a plain majority rule lets
    six monochrome-theme inversions through at confidence 0.95.)

    Rule 3 is the one that reads the REST of the board, and it is there
    because rule 2 is satisfied by more than pieces. The air above a
    near-topped-out stack is routinely tetromino-sized — four columns
    topping out at row 1 leave four cells of air over a stack grounded at
    row 0 — so inverted, it is piece-shaped and piece-sized and rule 2
    waves it through. What gives it away is everything else the inversion
    turns upside down: the stack's buried holes become cells hanging over
    the void, and there are many more of them than a piece could account
    for. Measured over 300 seeded legal near-top-out boards per style
    (six-plus columns grounded at row 0, no complete row, holes below):
    142 read WRONG above the gate at up to 0.97 confidence with the
    earlier per-column cell budget, 0 with these rules — and the family
    gained no exact reading either way, since 12 to 32 cells hang over
    the void in every one of those inversions against a budget of four.
    Availability is what keeps the budget off the raw airborne count: on
    360 synthetic frames of a legal board with a piece falling through
    row 0, a whole-board airborne budget vouched only 103 (a messy stack
    cut into floating bands by its own holes), where rules 2 and 3 vouch
    for 350.

    What is left is a frame whose true air above the stack is itself
    tetromino-sized and tetromino-shaped AND drains to the floor down a
    covered well, so that nothing else hangs. It is confined to the
    bootstrap path — :class:`GridClassifier`'s memory, once anchored,
    never consults this.
    """
    cols = int(occupancy.shape[1])
    observable = [c for c in range(cols) if (0, c) not in unobservable]
    if not observable:
        return False
    occupied = [c for c in observable if bool(occupancy[0, c])]
    if not occupied:
        return True
    if len(occupied) * 2 > len(observable):
        return False
    airborne, grounded = _support(occupancy, unobservable)
    if not all(bool(airborne[0, c]) for c in occupied):
        return False
    piece = _flood(airborne, [(0, occupied[0])])
    if int(piece.sum()) > _PIECE_CELLS or not all(bool(piece[0, c]) for c in occupied):
        return False
    return int(_over_the_void(airborne, grounded).sum()) <= _PIECE_CELLS


def cell_scores(
    image: NDArray[np.uint8],
    rows: int = DEFAULT_HEIGHT,
    cols: int = 10,
    margin: float = 0.25,
    unobservable_cells: frozenset[tuple[int, int]] | None = None,
) -> NDArray[np.float32]:
    """Per-cell occupancy score in [0, 1]: distance from the background.

    The background is estimated per frame as the per-channel median of the
    TOP ROW's OBSERVABLE cell colors — a gravity prior: the top row is
    usually background, contaminated by at most the 4 cells of a piece in
    flight, so the median is taken over a majority of true background
    cells. This recovers even boards that are mostly filled (where any
    dominant-cluster estimate would lock onto the pieces and invert the
    reading). ``unobservable_cells`` names cells whose pixels belong to a
    UI panel rather than the board; they are left out of the sample.

    The prior's limit: a STACK reaching the visible top row is legal,
    reachable Tetris (side columns stacked to row 0 while the spawn
    columns stay clear; versus-mode garbage rows, 9/10 filled, pushed up
    to the top) — it is NOT game over. With >= 6 of the 10 top-row cells
    non-background the median locks onto the pieces and the polarity of
    every score inverts. :func:`classify_grid` therefore refuses to vouch
    for a reading whose own top row is grounded in it (see
    :func:`_top_row_vouchable`); streaming callers should use
    :class:`GridClassifier`, whose committed background memory keeps such
    boards readable at full confidence.
    """
    colors = _cell_colors(image, rows, cols, margin)
    background = _top_row_background(colors, unobservable_cells or frozenset())
    return _distance_scores(colors, background)


def otsu_threshold(values: NDArray[np.float32]) -> float:
    """Exact Otsu threshold for a small set of continuous values.

    Evaluates every split between consecutive sorted unique values and
    returns the midpoint maximizing between-class variance.
    """
    vals = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = vals.size
    if n < 2:
        return float(vals[0]) if n else 0.5
    total = vals.sum()
    best_split = 0.5 * (vals[0] + vals[-1])
    best_var = -1.0
    cum = 0.0
    for i in range(n - 1):
        cum += vals[i]
        if vals[i + 1] == vals[i]:
            continue
        n0 = i + 1
        n1 = n - n0
        mu0 = cum / n0
        mu1 = (total - cum) / n1
        var = n0 * n1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_split = 0.5 * (vals[i] + vals[i + 1])
    return float(best_split)


def otsu_threshold_hist(values: NDArray[np.float32], bins: int = 256) -> float:
    """Histogram-based Otsu threshold for pixel-scale inputs.

    Same threshold as :func:`otsu_threshold` to within one bin width, in
    O(n + bins) numpy instead of a per-sample Python loop — use it when
    ``values`` are thousands of pixels. :func:`otsu_threshold` remains the
    exact path for small value sets (e.g. the rows x cols cell scores of a
    board).
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    n = int(vals.size)
    if n < 2:
        return float(vals[0]) if n else 0.5
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        return lo
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    weights = counts.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Split after bin i: class 0 = bins [0..i], class 1 = bins [i+1..].
    w0 = np.cumsum(weights)[:-1]
    w1 = float(n) - w0
    csum = np.cumsum(weights * centers)
    total = csum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = csum[:-1] / w0
        mu1 = (total - csum[:-1]) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    var = np.where((w0 > 0) & (w1 > 0), var, -1.0)
    # Well-separated classes leave a plateau of empty bins where the
    # criterion is bit-identical; split in its middle, mirroring the exact
    # variant's midpoint-of-the-gap threshold.
    near = np.flatnonzero(var >= var.max())
    return float(edges[int(near[0] + near[-1]) // 2 + 1])


def classify_grid(
    image: NDArray[np.uint8],
    rows: int = DEFAULT_HEIGHT,
    cols: int = 10,
    background: NDArray[np.float64] | tuple[float, ...] | None = None,
    unobservable_cells: frozenset[tuple[int, int]] | None = None,
) -> tuple[NDArray[np.bool_], float]:
    """Classify a board-region image into per-cell occupancy.

    Returns ``(occupancy, confidence)`` where ``occupancy`` is a
    (rows, cols) bool array (True = occupied) and ``confidence`` in [0, 1]
    reflects how cleanly the two classes separate. Confidence semantics:
    0.0 = unreadable or unvouchable, the caller's gate must reject; 0.5 =
    trusted uniform-empty; (0, 1] = class-gap fraction for a two-class
    frame.

    ``background`` is the board's known background color (same channel
    order and count as ``image``), e.g. from :class:`GridClassifier`'s
    cross-frame memory; when given, scores are distances from it and the
    top-row prior below is not consulted.

    ``unobservable_cells`` names ``(row, col)`` cells whose pixels are a UI
    panel's rather than the board's; they are excluded from the top-row
    background sample and from the top-row cap below. Omitted (the
    default), every behavior is exactly what it was.

    Without ``background`` the top-row-median estimate is used, under one
    cap: a two-class reading whose OWN top row contradicts the estimate's
    majority-background prior — because a majority of its observable cells
    read occupied, or because an occupied one is GROUNDED rather than a
    piece in flight — is the very configuration in which the median can
    lock onto piece colors and invert every cell at high apparent
    confidence (a legal, reachable state: side columns stacked to row 0,
    versus garbage pushed to the top; see :func:`cell_scores` and
    :func:`_top_row_vouchable`). Such readings keep their occupancy but
    are capped to confidence 0.0: without cross-frame memory their
    polarity cannot be vouched for, and a wrong reading above the gate is
    worse than a dropped frame. A piece merely spawning or falling through
    row 0 is NOT that configuration and keeps its confidence — pieces are
    visible in row 0 in most games, and capping those was what left a game
    whose preview also covers two top-row cells permanently unreadable.
    """
    colors = _cell_colors(image, rows, cols, margin=0.25)
    if background is not None:
        return _classify_scored(colors, np.asarray(background, dtype=np.float64))
    return _self_estimated(colors, unobservable_cells or frozenset())


def _self_estimated(
    colors: NDArray[np.float32],
    unobservable: frozenset[tuple[int, int]],
) -> tuple[NDArray[np.bool_], float]:
    """The memoryless path: top-row-median background, plus the top-row cap."""
    occupancy, confidence = _classify_scored(colors, _top_row_background(colors, unobservable))
    if confidence > 0.0 and not _top_row_vouchable(occupancy, unobservable):
        confidence = 0.0
    return occupancy, confidence


def _classify_scored(
    colors: NDArray[np.float32],
    background: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], float]:
    """Split per-cell colors into empty/occupied by distance from ``background``."""
    rows, cols = int(colors.shape[0]), int(colors.shape[1])
    scores = _distance_scores(colors, background)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi < MIN_SPREAD:
        # Uniform-near: every cell sits at the background estimate — an
        # EMPTY board, whatever color the theme paints it (a solid
        # near-white region is a light theme's empty board just as a
        # solid dark one is a dark theme's). Classified empty with usable
        # confidence so a board wipe (game over, new game) reaches the
        # tracker instead of being dropped at the gate. Under a
        # SELF-estimated background a whole-board flash or a solid pause
        # overlay reads the same way — the frames are
        # pixel-indistinguishable from an empty board — and the tracker's
        # reset debounce is what keeps short flashes from committing
        # anything; under a REMEMBERED background only a solid frame at
        # the board's own background color lands here, and a bright
        # overlay on a dark theme falls to uniform-far below.
        return np.zeros((rows, cols), dtype=np.bool_), _UNIFORM_EMPTY_CONFIDENCE
    if hi - lo < MIN_SPREAD:
        # Uniform-far: every cell is far from the background estimate yet
        # mutually similar — the estimator is inconsistent with the field
        # (a heterogeneous banner drawn across the top row over a solid
        # field; a solid pause panel or flash in a color that is NOT a
        # remembered background). Truly unreadable: described as
        # all-occupied at zero confidence so the caller's gate rejects it;
        # it is never classified as empty.
        return np.ones((rows, cols), dtype=np.bool_), 0.0

    threshold = otsu_threshold(scores)
    occupancy = scores > threshold
    empty_scores = scores[~occupancy]
    if empty_scores.size and float(empty_scores.max()) >= MIN_SPREAD:
        # Otsu split piece-vs-piece: on a nearly full board, a palette
        # with a near-background piece color (paper-white's lavender) can
        # put the between-class variance peak BETWEEN two piece clusters,
        # leaving cells far from the background inside the "empty" class.
        # A cell scoring below MIN_SPREAD is at the background by
        # construction (see the measured anchors on MIN_SPREAD), so
        # re-anchor the split at the floor; the class-gap confidence
        # below still rejects frames with no real gap around it (smooth
        # gradients and other continuums).
        occupancy = scores > MIN_SPREAD
        empty_scores = scores[~occupancy]
    occupied_scores = scores[occupancy]
    if occupied_scores.size == 0 or empty_scores.size == 0:
        return occupancy, 0.0
    # Fraction of the observed score range separating the two classes:
    # 1.0 when the split is wide open, near 0 when samples nearly touch.
    gap = float(occupied_scores.min() - empty_scores.max())
    confidence = float(np.clip(gap / (hi - lo), 0.0, 1.0))
    return occupancy, confidence


# Acceptance bar for GridClassifier's memory updates. Mirrors the default
# frame gate (CoachConfig.min_confidence in app.py, which passes its
# configured value in explicitly); kept as a literal here because the
# vision layer must not import the app layer.
_DEFAULT_MIN_CONFIDENCE = 0.15


class GridClassifier:
    """:func:`classify_grid` plus a cross-frame background-color memory.

    The per-frame top-row prior cannot vouch for a board whose stack
    legally reaches visible row 0, and a lone frame cannot tell a bright
    pause panel on a dark theme from a light theme's empty board. One
    remembered color — the board's background, re-measured from the
    empty class of every accepted frame — settles both: row-0-filled
    boards read exactly at full confidence, and a solid frame far from
    the remembered background is unreadable (all-occupied @0.0) rather
    than a phantom board wipe.

    Memory protocol (correctness over availability, per the module
    invariant):

    - No memory yet: frames classify with the self-estimated top-row
      prior, top-row cap included. An accepted two-class frame anchors the
      memory on its empty-class median and CONFIRMS it; an accepted
      uniform-empty frame anchors it provisionally (the frame's overall
      median) — a solid frame alone cannot prove it is the board.
    - Provisional memory: used first; if the frame is rejected under it,
      the frame re-bootstraps (a wrong provisional anchor, e.g. a pause
      panel covering the board at startup, must not blind the session).
      The first accepted two-class frame confirms the memory.
    - Confirmed memory: never falls back — a frame it cannot read is
      rejected, holding the last committed state. This is deliberate: the
      plausible-looking alternatives (a top-row re-estimate on a garbage
      board, a bright overlay re-read as a light-theme empty board) are
      exactly the wrong readings the memory exists to prevent. It tracks
      slow drift, re-measuring from every accepted frame.
    - Confirmed but not yet CORROBORATED: one frame is one frame. Until a
      second, independent one has backed the anchor up, every ACCEPTED
      frame gets a second opinion (:meth:`_second_opinion`) — the same
      frame classified from scratch by the top-row prior, which knows
      nothing of the memory. Agreement corroborates the anchor and ends
      the checks for the session; a vouched disagreement means one of the
      two is inverted and there is no way to tell which, so the memory is
      dropped and the frame with it, and the next frame bootstraps fresh.
      Frames that offer no opinion (rejected by the prior, or uniform)
      leave the anchor as it is, so a game whose top row is never
      vouchable still runs on a confirmed, never-corroborated memory.

    - Impossible readings, at EVERY stage, corroborated included: a
      frame the anchor reads as a board containing a completed row
      (:func:`_claims_a_completed_row`) is refused, and the anchor is
      asked to justify itself (:meth:`_impossible_frame`).

    The last two stages both exist because a legal near-top-out board CAN
    be read inverted and vouched for (see :func:`_top_row_vouchable`), and
    such a reading anchors a CONFIRMED memory on the PIECE color; a
    confirmed memory never falls back, so every frame after it is wrong.
    Measured on the ROAS Stacker geometry, cream-mono, with neither stage
    in place: one covered-well bootstrap frame anchored at
    (150, 140, 120) instead of (245, 240, 228) and the whole session read
    116 of 116 observable cells wrong at confidence 0.60.

    Corroboration alone does not close that, because two frames are
    independent as CAPTURES but not as BOARDS. The shape that inverts is
    a stack, and a stack persists: at 15 fps the same near-top-out board
    is captured every 67 ms, and frame two's own inverted reading
    corroborates frame one's wrong anchor. (Measured: holding that board
    for two frames — or playing five different boards of the same family
    — corroborates the piece-color anchor and wedges the session
    permanently, which is what the impossible-reading stage was added
    for.)

    What the inversion cannot survive is an ORDINARY frame. Inverted, the
    empty air above a normal stack reads as row after row occupied end to
    end — completed lines, which clear the instant they complete and so
    can never be on screen. That contradiction needs no prior and no
    second frame, and it is loudest on exactly the frames a
    near-top-out bootstrap lacks.

    ``unobservable_cells`` is the same set the engine computes from the two
    selected rectangles. It matters most HERE: a covered top-row cell is
    occupied on every frame of the session, so before this the cap fired
    on every frame, nothing was ever accepted, and the memory that exists
    to rescue exactly these boards could never anchor — a deadlock, not a
    degradation.

    Single-threaded use only (the engine's worker thread).
    """

    def __init__(
        self,
        rows: int = DEFAULT_HEIGHT,
        cols: int = 10,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        unobservable_cells: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._min_confidence = min_confidence
        self._unobservable: frozenset[tuple[int, int]] = unobservable_cells or frozenset()
        self._background: NDArray[np.float64] | None = None
        self._confirmed = False
        self._corroborated = False
        self._impossible = 0

    @property
    def background(self) -> NDArray[np.float64] | None:
        """Remembered background color; None before any anchor."""
        return None if self._background is None else self._background.copy()

    @property
    def confirmed(self) -> bool:
        """True once an accepted two-class frame anchored the memory."""
        return self._confirmed

    @property
    def corroborated(self) -> bool:
        """True once a second, independent frame backed the anchor up."""
        return self._corroborated

    def classify(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.bool_], float]:
        """Classify one frame; same return contract as :func:`classify_grid`."""
        colors = _cell_colors(image, self._rows, self._cols, margin=0.25)
        if self._background is not None:
            occupancy, confidence = _classify_scored(colors, self._background)
            if confidence >= self._min_confidence:
                if self._confirmed and not self._corroborated:
                    agrees = self._second_opinion(colors)
                    if agrees is False:
                        # An anchor and a from-scratch reading of the same
                        # frame that disagree about the background cannot
                        # both be the board: believe neither.
                        self._forget()
                        return occupancy, 0.0
                    self._corroborated = agrees is True
                if _claims_a_completed_row(occupancy, self._unobservable):
                    # The anchor is describing a board that cannot exist.
                    return occupancy, self._impossible_frame(colors)
                self._remember(colors, occupancy)
                return occupancy, confidence
            if self._confirmed:
                return occupancy, confidence
            # Provisional memory failed on this frame: fall through and
            # re-bootstrap, so a wrong provisional anchor cannot wedge
            # the session before play has even been observed.
        occupancy, confidence = self._bootstrap(colors)
        if confidence >= self._min_confidence:
            self._remember(colors, occupancy)
        return occupancy, confidence

    def _bootstrap(self, colors: NDArray[np.float32]) -> tuple[NDArray[np.bool_], float]:
        """Self-estimated classification: classify_grid's memoryless path."""
        return _self_estimated(colors, self._unobservable)

    def _remember(self, colors: NDArray[np.float32], occupancy: NDArray[np.bool_]) -> None:
        """Re-measure the background from an accepted frame.

        Unobservable cells are left out of every sample: their pixels are
        the covering panel's, and a memory measured partly from the panel
        would drift toward a color the board never shows.

        Reaching here is also what ends a run of impossible frames: this
        is called exactly when a frame was accepted AND describes a board
        that can exist, on the bootstrap path as well as the remembered
        one, so no count survives into the next anchor.
        """
        self._impossible = 0
        observable = self._observable_mask(occupancy.shape)
        if bool((occupancy & observable).any()):
            # Two-class frame: the empty class IS the background, freshly
            # measured. Anchoring here is what makes polarity survive the
            # stack growing into row 0.
            measured = self._empty_class_color(colors, occupancy, observable)
            if measured is None:  # every observable cell reads occupied
                return
            self._background = measured
            self._confirmed = True
        else:
            # Uniform-empty frame: the whole frame is the background —
            # but only provisionally, since a solid pause panel is
            # indistinguishable from an empty board (never CONFIRM from
            # one, or a panel at startup would poison the session).
            flat = colors.reshape(-1, colors.shape[-1])
            keep = observable.ravel()
            sample = flat[keep] if keep.any() else flat
            self._background = np.asarray(np.median(sample, axis=0), dtype=np.float64)

    def _second_opinion(self, colors: NDArray[np.float32]) -> bool | None:
        """Does a from-scratch reading of this frame back the anchor up?

        Classifies the frame by the top-row prior alone — no memory in it
        — and compares the background THAT reading implies against the
        remembered one. True when they are the same color, False when a
        vouched reading names a different one, None when the frame has no
        opinion to offer: refused by the prior (:func:`_top_row_vouchable`
        would not vouch for it, so it is no evidence about anything), or
        uniform-empty (a solid frame cannot tell the board's background
        from a panel's, which is why such frames only ever anchor
        provisionally).

        "The same color" is the module's own floor for it: colors closer
        than :data:`MIN_SPREAD` on the score scale are what a single cell
        may not be split on either, which leaves room for the slow drift
        the memory tracks (~54 uint8 units of Euclidean distance) while an
        inverted anchor sits far outside it (the piece-vs-paper case
        measured above: 0.63).
        """
        background = self._background
        if background is None:
            return None
        occupancy, confidence = self._bootstrap(colors)
        if confidence < self._min_confidence:
            return None
        observable = self._observable_mask(occupancy.shape)
        if not bool((occupancy & observable).any()):
            return None
        measured = self._empty_class_color(colors, occupancy, observable)
        if measured is None:
            return None
        return float(_distance_scores(measured, background)) < MIN_SPREAD

    def _impossible_frame(self, colors: NDArray[np.float32]) -> float:
        """Refuse a frame the anchor read as an impossible board.

        Always reports confidence 0.0: whatever the split looked like, a
        reading that claims a completed row is not a board, and
        committing it would be worse than holding the last one.

        Whether the MEMORY survives is the real question, and the frame
        is asked the same way a bootstrap-stage frame is — by second
        opinion (:meth:`_second_opinion`). An ordinary play frame under
        an inverted anchor answers it immediately: its own top row is
        clean background, so the top-row prior reads it from scratch,
        vouches, and names a different color — the anchor is inverted and
        goes at once, unlike the corroboration stage this runs FOR THE
        LIFE OF THE SESSION.

        A frame with no second opinion to give (the prior refuses a
        board too full to read, or the frame is uniform) only counts
        against the anchor: a correct one can produce a stray impossible
        frame — a line clear caught while the row is still lit, a
        game-over fill — and is worth far more than the frames it costs
        to be sure, while an inverted one produces them without end. The
        memory is dropped once the contradiction has persisted
        (:data:`_IMPOSSIBLE_FRAME_LIMIT`).
        """
        if self._second_opinion(colors) is False:
            self._forget()
            return 0.0
        self._impossible += 1
        if self._impossible >= _IMPOSSIBLE_FRAME_LIMIT:
            self._forget()
        return 0.0

    def _forget(self) -> None:
        """Drop the memory back to nothing, so the next frame bootstraps."""
        self._background = None
        self._confirmed = False
        self._corroborated = False
        self._impossible = 0

    @staticmethod
    def _empty_class_color(
        colors: NDArray[np.float32],
        occupancy: NDArray[np.bool_],
        observable: NDArray[np.bool_],
    ) -> NDArray[np.float64] | None:
        """Median color of the OBSERVABLE cells a reading calls empty."""
        flat = colors.reshape(-1, colors.shape[-1])
        sample = flat[(~occupancy.ravel()) & observable.ravel()]
        if sample.size == 0:
            return None
        return np.asarray(np.median(sample, axis=0), dtype=np.float64)

    def _observable_mask(self, shape: tuple[int, ...]) -> NDArray[np.bool_]:
        """(rows, cols) True wherever the capture really shows the board."""
        mask = np.ones(shape, dtype=np.bool_)
        rows, cols = int(shape[0]), int(shape[1])
        for r, c in self._unobservable:
            if 0 <= r < rows and 0 <= c < cols:
                mask[r, c] = False
        return mask

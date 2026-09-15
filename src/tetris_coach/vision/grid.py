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
excluded from the top-row background sample, from the top-row cap and
from the ghost layer below — otherwise a permanently covered top-row
cell is a permanently "occupied" top row, the cap fires on every frame,
and :class:`GridClassifier`'s memory (which only anchors from ACCEPTED
frames) can never form. The default, an empty set, leaves every
behavior exactly as it was.

A two-way split is not always the whole story. Two things put a
translucent THIRD level on the board, scoring between the background and
a real piece, and Otsu has to give it to one side or the other — which
side varying frame to frame. :func:`_own_paint_layer` names the first:
this tool's OWN placement hint, which is on screen when the capture is
taken, so the coach reads its own overlay back as board content. It is
recognized by its color, which is known in advance. :func:`_ghost_layer`
names the second: a game-drawn landing preview, which has to be argued
for from structure. See each for the rule, what it costs, and which way
it errs.


Channel order does not matter (BGR vs RGB): Euclidean distance and the
per-channel median are permutation-equivariant in the channel axis.
A 2-D grayscale image is simply the single-channel (C=1) case.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT
from ..core.pieces import ROTATIONS

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

# Ceiling on what a frame carrying a named landing preview may report, and
# deliberately the same value as above, for the same reason: such a frame's
# occupancy does not follow from the gap the confidence measures. The gap
# is taken with the layer's cells out of both classes, so a frame that
# genuinely has THREE levels reports the separation of two of them —
# without this, one reads like a textbook frame (measured on a synthetic
# three-level board: 0.94). Whether those cells are a preview or a real
# pale piece is settled by :func:`_ghost_layer`'s structure, so the
# reading is structurally grounded exactly as a uniform-empty one is, and
# says so with the same number.
#
# What this is NOT is a second chance to catch a wrongly named layer, and
# no threshold here could be. Measured: a synthetic board where the rule
# deletes real content reads 0.94 named and 0.58 with the rule off — both
# far above any usable gate — while the conservative measure that WOULD
# flag it (count the layer in the empty class, i.e. report the gap the
# frame has without the rule) reads 0.078 on the fifteen frames of
# live_session where the layer is a real ghost, which is the pre-rule
# number that had them rejected. The rule itself is the defense; this
# only keeps a rule-dependent reading from wearing a clean frame's
# signature, and leaves a user who tightens the gate past it a way to
# take only frames that need no rule at all.
_LAYER_CONFIDENCE_CEILING = 0.5

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

# How far clear of the background cluster an intermediate score level must
# stand before it is believed to be a rendering LAYER of its own rather
# than the background's own noise. Half the floor, because the layer being
# looked for is by definition NOT a different color from the background
# (that is what scoring below :data:`MIN_SPREAD` means) while the
# background's own spread is an order of magnitude smaller: measured on
# the ghost session, the background cluster spans 0.00-0.02 and the ghost
# sits at 0.32, a clearance of 0.30 and 0.21 on the two frames where the
# panel's own cells raise the background class. Doubles as the band's
# lower edge, so a cell nearer the background than this is simply
# background and is never named a ghost.
_GHOST_SEPARATION = MIN_SPREAD / 2

# The largest share of the OBSERVABLE playfield a candidate band may claim
# as content before it is read as a second GROUND instead. See
# :func:`_content_budget` for the argument and the measurement.
_BAND_CONTENT_SHARE = 0.25

# Opacity of the fill this tool paints its own placement hint with. THE
# one definition: overlay.renderer.HintStyle draws with it and
# :func:`_own_paint_layer` recognizes the result, so the painter and the
# reader cannot drift apart. It lives here rather than in overlay/
# because vision/ must stay importable without a display, and overlay/
# may not (PySide6).
HINT_FILL_OPACITY = 0.18

# How near a cell's color must sit to the composite the tool's own hint
# would produce there before that cell is called the tool's own paint
# rather than the board. Plain Euclidean distance in uint8 units, on the
# same scale the colors are. Measured over every observable cell of all
# four committed session windows plus the roas_stacker frames (39040
# cells, 309 of them under the hint):
#
#   the hint fill over the BOARD's own ground   0.470 - 0.565  (272 cells)
#   the hint fill over a REAL PIECE            58.41 - 240.63  (37 cells)
#   the nearest cell with no hint on it at all         23.77
#
# Three clusters, not two, and the middle one is the point: the same
# translucent fill over a piece composites to a different color, so a
# hint drawn on top of real content does not match and those cells are
# never taken out (live_session 59-62, where the I hard-drops under the
# hint that was pointing at its landing square, and the badge cells
# below). What the rule deletes is cells that are EMPTY BOARD WITH OUR
# PAINT ON THEM, which is the only thing it can safely claim.
#
# The nearest non-hint cell is the pale periwinkle piece itself — the one
# color in these fixtures that must never be deleted (absorbed_piece
# frame 280 at (0,3), scoring 0.346). 8.0 is 14x above the first cluster
# and 3x below the third, which leaves room for a rescaled capture's
# resampling without coming near a real piece color.
_OWN_PAINT_TOLERANCE = 8.0

# Every tetromino rotation as a normalized (row, col) cell tuple, mapped to
# the piece it belongs to. No two pieces share a rotation, so a cell set
# names at most one of them. A ghost is a copy of ONE piece, so an
# intermediate layer that is not the shape of a piece is not a ghost, and
# WHICH piece it is, is what ties the layer to the falling piece it
# previews (see :func:`_ghost_layer`).
_SHAPE_PIECES: dict[tuple[tuple[int, int], ...], str] = {
    tuple(sorted(rotation.cells)): piece
    for piece, rotations in ROTATIONS.items()
    for rotation in rotations
}


class OwnPaint(NamedTuple):
    """The color this tool paints its own placement hint with.

    The coach draws its hint ON TOP of the game and then captures the
    screen again, so its own overlay comes back round as input. That is
    the one piece of furniture on the board whose exact appearance is
    known in advance rather than guessed at: a translucent fill of a
    single configured color, composited over whatever the board shows
    underneath. :func:`_own_paint_layer` recognizes it from that.

    ``color`` is the hint color in the SAME channel order as the frames
    the classifier is handed — the capture pipeline produces BGR, which
    is what :data:`HINT_PAINT` holds. Nothing else in this module depends
    on channel order; this does, unavoidably, because it is the only rule
    here that knows a specific color rather than a distance. Handed
    frames in the other order (or grayscale) the paint simply never
    matches, and the reading falls back to :func:`_ghost_layer`'s
    structural rule — a degradation, not a failure.

    ``opacity`` is the alpha the fill is drawn at and ``tolerance`` how
    far a cell may sit from the resulting composite and still be called
    paint (see :data:`_OWN_PAINT_TOLERANCE`).
    """

    color: tuple[float, ...]
    opacity: float = HINT_FILL_OPACITY
    tolerance: float = _OWN_PAINT_TOLERANCE

    @classmethod
    def for_hint_color(
        cls,
        color: str,
        opacity: float = HINT_FILL_OPACITY,
        tolerance: float = _OWN_PAINT_TOLERANCE,
    ) -> OwnPaint | None:
        """A hint color as BGR, or ``None`` when it is not a hex color.

        ``cli.py`` takes ``--hint-color`` as any Qt-parsable color string,
        which includes names ("cyan") and forms this module has no parser
        for. Those turn the rule OFF rather than raising: the session
        still runs, on :func:`_ghost_layer`'s structural rule alone, which
        is exactly where it stood before this rule existed. Returning
        ``None`` for a color the painter accepts is the honest answer —
        guessing at the paint is how a rule that deletes cells goes wrong.
        """
        text = color.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
            return None
        r, g, b = (float(int(text[i : i + 2], 16)) for i in (0, 2, 4))
        return cls((b, g, r), opacity, tolerance)


# The default hint paint: app.CoachConfig.hint_color at HINT_FILL_OPACITY,
# in the capture pipeline's BGR order. app.CoachEngine passes its own
# configured color in; this default keeps a bare classify_grid call
# recognizing the overlay the tool actually ships with.
HINT_PAINT = OwnPaint.for_hint_color("#00e5ff")


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


def _observable(
    shape: tuple[int, int], unobservable: frozenset[tuple[int, int]]
) -> NDArray[np.bool_]:
    """(rows, cols) True wherever the capture really shows the board."""
    mask = np.ones(shape, dtype=np.bool_)
    rows, cols = int(shape[0]), int(shape[1])
    for r, c in unobservable:
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = False
    return mask


def _clearance_above(
    scores: NDArray[np.float32],
    cells: NDArray[np.bool_],
    observable: NDArray[np.bool_],
) -> float:
    """How far the next level up stands clear of ``cells``' own level.

    The other half of :func:`_clear_of_background`, and the half that
    tells a LEVEL from one slice of a RAMP. A level has air on both
    sides; a lighting gradient has cells every few hundredths all the way
    up, so whatever slice of it falls inside the band has the rest of the
    ramp sitting right on top of it (measured on a paper-white board
    under a 140-unit vertical gradient: 0.074 above, against 0.469 for
    the pale periwinkle T of ``tests/fixtures/pale_piece`` and 0.30 for a
    ghost).

    ``inf`` when there is nothing above at all — a ceiling that is not
    there constrains nothing.
    """
    if not bool(cells.any()):
        return float("inf")
    level = float(scores[cells].max())
    above = scores[(scores > level) & observable]
    return float(above.min()) - level if above.size else float("inf")


def _clear_of_background(
    scores: NDArray[np.float32],
    cells: NDArray[np.bool_],
    observable: NDArray[np.bool_],
) -> bool:
    """Does ``cells``' score level stand clear of the background cluster?

    THE test that separates a rendering LAYER of its own — a translucent
    preview, a pale-on-pale piece — from the background's own noise, and
    the one thing every reading of the intermediate band is conditioned
    on. It is relative rather than absolute because the background's
    spread is a property of the theme and the capture, not of the scale:
    measured, an ordinary background cluster spans 0.00-0.02 while the
    levels this module has to name sit at 0.32-0.35, a clearance of
    0.30-0.33.

    A CONTINUUM fails it, which is the point. The game's own start screen
    (``tests/fixtures/roas_stacker/start_screen_board.png``) is text
    antialiased over white: its faintest sub-floor cells run 0.174 up to
    0.196 with no gap anywhere, a clearance of 0.022. Nothing there is a
    layer and nothing there is content, and a rule that read it either
    way would find furniture on every start screen.

    Nothing below the level at all is a fail too, not a pass: a band with
    no background under it is not standing clear of anything.
    """
    if not bool(cells.any()):
        return False
    level = float(scores[cells].min())
    below = scores[(scores < level) & observable]
    return below.size > 0 and level - float(below.max()) >= _GHOST_SEPARATION


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


def _piece_named(cells: list[tuple[int, int]]) -> str | None:
    """Which tetromino ``cells`` are, or ``None`` when they are not one."""
    if len(cells) != _PIECE_CELLS:
        return None
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return _SHAPE_PIECES.get(tuple(sorted((r - min_r, c - min_c) for r, c in cells)))


def _pieces_in_flight(
    solid: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
) -> set[str]:
    """Names of the tetrominoes standing whole and airborne on this board.

    Airborne in :func:`_support`'s sense — not 4-connected to the floor
    through occupied cells — so a piece that has joined the stack is not
    in flight, and a fragment the top edge or a UI panel has cut in half
    is not a whole tetromino and names nothing.
    """
    airborne, _grounded = _support(solid, unobservable)
    names: set[str] = set()
    seen = np.zeros_like(airborne)
    for row, col in zip(*np.nonzero(airborne), strict=True):
        if seen[row, col]:
            continue
        component = _flood(airborne, [(int(row), int(col))])
        seen |= component
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(component), strict=True)]
        name = _piece_named(cells)
        if name is not None:
            names.add(name)
    return names


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
    own_paint: OwnPaint | None = HINT_PAINT,
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
    unobservable = unobservable_cells or frozenset()
    if background is not None:
        reading = _classify_scored(
            colors, np.asarray(background, dtype=np.float64), unobservable, own_paint
        )
        return reading.occupancy, reading.confidence
    reading = _self_estimated(colors, unobservable, own_paint)
    return reading.occupancy, reading.confidence


class _Reading(NamedTuple):
    """One frame's classification, plus the layer that is not board content.

    ``ghost`` is the (rows, cols) mask of cells :func:`_ghost_layer` named
    as a landing preview, or ``None`` when the frame had no third level.
    Those cells are already EMPTY in ``occupancy``; the mask is carried
    separately because they are not evidence about the BACKGROUND either
    — :class:`GridClassifier` re-measures its memory from the empty class
    and must not average a translucent overlay into it.
    """

    occupancy: NDArray[np.bool_]
    confidence: float
    ghost: NDArray[np.bool_] | None


def _self_estimated(
    colors: NDArray[np.float32],
    unobservable: frozenset[tuple[int, int]],
    own_paint: OwnPaint | None = HINT_PAINT,
) -> _Reading:
    """The memoryless path: top-row-median background, plus the top-row cap."""
    reading = _classify_scored(
        colors, _top_row_background(colors, unobservable), unobservable, own_paint
    )
    if reading.confidence > 0.0 and not _top_row_vouchable(reading.occupancy, unobservable):
        return reading._replace(confidence=0.0)
    return reading


def _own_paint_cells(
    colors: NDArray[np.float32],
    background: NDArray[np.float64],
    unobservable: frozenset[tuple[int, int]],
    paint: OwnPaint | None,
) -> NDArray[np.bool_] | None:
    """Cells whose color IS the composite this tool's hint fill makes here.

    The measurement alone, with none of :func:`_own_paint_layer`'s
    structural tests on top: ``None`` when there is no paint to look for
    (no configured color, or one in the wrong channel count), otherwise
    the mask of cells within :attr:`OwnPaint.tolerance` of
    ``background + opacity * (paint - background)``.

    :func:`_own_paint_layer` is the rule that DELETES those cells and can
    refuse to. This is the raw finding, which the split needs even when
    the rule refuses: a cell that looks like our own paint is not board
    content, so it may never be promoted out of the intermediate band as
    if it were (see :func:`_classify_scored`).
    """
    if paint is None:
        return None
    if len(paint.color) != int(colors.shape[-1]):
        return None
    bg = np.asarray(background, dtype=np.float64)
    expected = bg + paint.opacity * (np.asarray(paint.color, dtype=np.float64) - bg)
    diff = np.asarray(colors, dtype=np.float64) - expected
    painted = np.sqrt(np.sum(diff * diff, axis=-1)) <= paint.tolerance
    return painted & _observable((int(colors.shape[0]), int(colors.shape[1])), unobservable)


def _own_paint_layer(
    colors: NDArray[np.float32],
    background: NDArray[np.float64],
    solid: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
    paint: OwnPaint | None,
) -> NDArray[np.bool_] | None:
    """The cells carrying the coach's OWN hint overlay, or ``None``.

    This tool paints a placement hint over the game and then captures the
    screen again, so the hint is in its own next frame. Read as board
    content it is a tetromino that appears from nowhere, sits exactly
    where a piece would come to rest, and then TELEPORTS the moment the
    solver changes its mind — which is a phantom lock, an unexplainable
    frame, and a re-solve that moves the hint again. That loop is what
    every committed session window in ``tests/fixtures`` actually
    contains, and the evidence is in the pixels: on
    ``ghost_beside_stack`` frame 147 the cell at (11, 0) is 1560 px of
    ``(206, 248, 253)`` ringed by 442 px of exactly ``(0, 229, 255)`` —
    ``HintStyle.color`` ``#00e5ff`` for the pen at full opacity, and the
    same color at :data:`HINT_FILL_OPACITY` over that board's own
    ``(251, 252, 252)`` for the fill (RGB; the frames arrive BGR).
    The one-cell round badge the SPEC used to attribute to the game is
    this module's own rotation badge, drawn by
    ``overlay.renderer._draw_rotation_badge`` above the hint's top-left
    cell and carrying the rotation index as its digit.

    Which means this game — ROAS Stacker, the only game any committed
    fixture holds — draws NO GHOST AT ALL. Measured over all four
    windows: of the 582 cells that ever land in :func:`_ghost_layer`'s
    band, 272 are this overlay and the other 310 are the real pale
    periwinkle T of ``absorbed_piece``; not one is a landing preview.
    Both halves of what the ghost rule was built from — the intermediate
    level that flickers, and the badge that floats over it — are the
    coach's own paint coming back round.

    Being the tool's own paint, it is the one layer on the board that
    does not have to be guessed at from structure. The fill is a straight
    alpha composite, so the color it produces over a known background is
    known too::

        expected = background + opacity * (paint - background)

    and a cell within :data:`_OWN_PAINT_TOLERANCE` of that is the
    overlay. Measured over every observable cell of all four committed
    windows plus the roas_stacker frames (39040 cells), that separates
    three clusters and not two: the fill over the board's own ground at
    0.470-0.565 (272 cells), the fill over a REAL PIECE at 58.41-240.63
    (37 cells), and the nearest cell with no hint on it at all at 23.77.

    The middle cluster is what makes the rule safe rather than merely
    accurate. The same translucent fill over a piece composites to a
    different color, so a hint drawn ON TOP of real content does not
    match and those cells are never taken out — ``live_session`` 59-62,
    where the I hard-drops onto the very square the hint was pointing
    at, is that case on real pixels. What this rule deletes is only
    cells that are EMPTY BOARD WITH OUR PAINT ON THEM, which is the only
    thing the color can honestly claim.

    And the far cluster is the pale periwinkle piece, the exact color
    :func:`_ghost_layer` spends five structural tests protecting.
    Nothing here can delete it: it is a different color, 3x outside the
    tolerance, and this rule looks at nothing else.

    That is also why this rule, unlike :func:`_ghost_layer`'s, needs no
    score band — and why it must not have one. On the fixtures' near-white
    ground the fill scores 0.321, inside the band; on a BLACK ground the
    same composite scores 0.374, above :data:`MIN_SPREAD` entirely, so it
    reads as an ordinary solid piece and a band-limited rule could not see
    it at all. Over grey grounds the score runs 0.284-0.374, crossing both
    edges of the band. The theme decides where the paint lands, which is
    precisely the thing a rule keyed to the paint itself does not care
    about.

    Two structural tests remain, and both are about what is around the
    paint rather than what it is:

    1. It must be exactly one tetromino. That is what the overlay draws.
       It is the guard for a partial match — a hint the panel mask cuts
       in half, or one lying half over real content — which refuses the
       whole widget rather than deleting a fragment. (No committed frame
       is a partial match: every frame that carries a hint matches either
       4 cells or 0.)
    2. Nothing solid may sit directly ABOVE it. The hint marks a hard
       drop's landing square, and a piece reaches one by falling down
       its own columns, so every cell above a hint cell is empty by
       construction — except for one thing, which is this tool's own
       rotation badge. The badge is paint too, but it is drawn OPAQUE
       over a single cell and blended with a black digit, so it matches
       no composite and nothing can name it (measured: 58-64 from the
       fill composite, and 0.49 on the score scale — a full piece
       color). Taking the hint out from under it would leave the badge
       behind as an unexplainable added cell: measured before any of
       this existed at four UNEXPLAINED frames, a BOARD_RESET, and the
       badge committed to the stack. Refusing the whole widget instead
       leaves those frames reading exactly as they always did — below
       the gate, last hint held. (The badge rides only a non-zero
       rotation: ``live_session`` 121-135, ``ghost_session`` 150 and
       ``absorbed_piece`` 298-301 and 360 are the frames that carry one.)

    The real fix for a tool reading its own output is for the capture
    never to contain the overlay in the first place; that is
    platform-specific window-exclusion work in ``capture/``, and this is
    the layer that keeps the reading correct whether or not it lands.
    """
    painted = _own_paint_cells(colors, background, unobservable, paint)
    if painted is None:
        return None
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(painted), strict=True)]
    # 1. Exactly one tetromino.
    if _piece_named(cells) is None:
        return None
    # 2. Nothing solid on top of it: the rotation badge rides there, and
    #    it is paint nothing can name.
    for r, c in cells:
        if r > 0 and bool(solid[r - 1, c]) and not bool(painted[r - 1, c]):
            return None
    return painted


def _ghost_layer(
    scores: NDArray[np.float32],
    unobservable: frozenset[tuple[int, int]],
) -> NDArray[np.bool_] | None:
    """The cells of a landing preview, or ``None`` when there is no such layer.

    Almost every modern Tetris draws a GHOST under the falling piece: a
    translucent copy of it, resting where a hard drop would put it. Being
    translucent, its cells are a blend of a piece color and the board's
    background, so they score BETWEEN the two — a third level in a
    measure built to carry two. Otsu must put that level on one side or
    the other, and on the session this was diagnosed from
    (``tests/fixtures/ghost_session``: background 0.00-0.02, ghost 0.32,
    solid piece 0.57-0.82) it lands on either side depending on what else
    is on the board. Read as content the ghost is a tetromino that
    teleports between frames, which is a phantom lock, then four
    unexplainable frames, then a BOARD_RESET that swallows the real
    falling piece; read as background it is merely invisible. It is not
    board content either way: nothing is there.

    A cell is a CANDIDATE for the layer when it scores in the band
    ``[_GHOST_SEPARATION, MIN_SPREAD)``: clear of the background, yet
    nearer to it than the module's own floor for "a different color".

    The band is only ever a candidate, never the answer, and the reason
    is measured rather than theoretical. This same game has a real piece
    color INSIDE the band — a pale periwinkle scoring 0.346 against its
    near-white ground, four thousandths under the floor, against the
    ghost's 0.320 (``tests/fixtures/roas_stacker``: the piece is on the
    stack at rows 10-11 of ``live2_board_00500`` and ``00600`` a hundred
    ticks apart, and is caught in mid-air, unambiguously a real falling
    piece, at rows 0-2 of ``00800`` and rows 1-2 of ``live_board_800``).
    No threshold on this scale separates 0.320 from 0.346, so nothing
    about a cell's score can decide this; only where the cells SIT can.

    And it must be decided the careful way round, because the costs are
    not symmetric: a ghost left in reads as a tetromino that teleports,
    which costs held frames, while real content read as a ghost DELETES
    STACK and hands the solver a board with room in it that does not
    exist. So the candidate is dropped unless the whole structure of a
    landing preview is there, and every one of these tests errs toward
    leaving the cells alone:

    1. It must stand clear of the background cluster by at least
       :data:`_GHOST_SEPARATION`. A level that blends into the background
       is the background, and inventing a layer inside it would widen the
       reported confidence of every ordinary frame.
    2. It must be exactly one tetromino: ``_PIECE_CELLS`` cells whose
       normalized shape is a real rotation (which also makes it
       4-connected — every tetromino is). A preview is a copy of ONE
       piece; a half-dozen scattered faint cells are something else and
       are left as content.
    3. It must REST — a landing preview is by definition where the piece
       comes to rest, so at least one of its cells must have the floor,
       or a cell at a real piece color, directly beneath it. A layer
       floating in mid-air is not a landing preview.
    4. Nothing may sit BESIDE or ON TOP of it: no cell of the layer may
       touch a solid cell to its left, to its right, or above. This is
       the test that keeps real content, and it is the one the band
       cannot do. A preview marks space the falling piece can still drop
       into, so it is the topmost thing in its own cells, with open air
       either side; a piece the stack has grown AROUND, or that carries
       anything on its shoulders, got there by being played. On the pale
       periwinkle above it fires on both frames: at ``00500`` the stack
       continues straight into the piece's left edge along the floor, and
       at ``00600`` there is stack directly over two of its cells.
       Support from BELOW is the one direction that stays legal — that is
       what test 3 requires — so a ghost resting on a flat stack surface
       is still named. A ghost resting inside a WELL is not: its sides
       touch, and the rule declines rather than guesses.
       Any solid neighbour refuses, not merely a grounded one, and the
       reason is a second piece of furniture this game draws: a little
       round "1" badge that floats directly over the preview (live
       session frames 121-135, ghost session frame 150). The badge is
       not board content either, but it is one cell and scores 0.49 —
       solidly a piece color — so nothing here can name it, and naming
       the preview under it leaves the badge behind as an unexplainable
       added cell. Measured: it costs four UNEXPLAINED frames, a
       BOARD_RESET, and the badge committed to the stack as a phantom
       floating cell. Refusing the whole widget instead leaves those
       frames reading exactly as they did before this rule existed —
       below the gate, hint held — which is the outcome to prefer.
    5. It must be a preview OF SOMETHING: the piece it copies has to be
       on the board, in flight. A whole tetromino of the SAME PIECE TYPE
       as the layer must stand airborne somewhere — airborne in
       :func:`_support`'s sense, not 4-connected to the floor, which is
       what a falling piece is and what a piece that has landed is not.
       This is the test that makes the residual error below rare instead
       of routine, and it is the one piece of structure a landing
       preview cannot be without: a game draws a ghost because a piece
       is falling, so no falling piece, no ghost. It also subsumes the
       old requirement that there be a solid class at all (a piece in
       flight IS one; with nothing but background and the band the frame
       is uniform-near and already reads empty).
       On the real pixels it is exactly the test that separates the two
       cases this function exists to tell apart, which is why it is
       worth the frames it costs: every ghost in both committed sessions
       has its own piece in the air above it (measured: 21 of 21 named
       frames — ghost_session 59-64, an O layer under a falling O;
       live_session 44-58, an I layer under a falling I), while on
       ``live2_board_00500`` the pale periwinkle T that must NOT be
       deleted sits under a falling J. Nothing about the periwinkle's
       score says which it is (0.346 against the ghost's 0.320) and, as
       the fixtures show, nothing about where it sits reliably does
       either — test 4 happens to refuse it only because the stack runs
       into its left edge on that frame, which one legal board
       difference removes.

    What is deliberately NOT required is the tighter structural story —
    same ROTATION as the falling piece, in the same columns, directly
    below it. It does not survive contact with the fixtures, and the
    reason is now known: what those frames carry is not a ghost but this
    tool's own hint (see :func:`_own_paint_layer`), which marks where the
    SOLVER wants the piece rather than where the player is holding it. On
    ghost_session 61-64 it sits at cols 2-3 while the O it belongs to is
    at cols 4-5, and on frame 150 it is a VERTICAL I at col 9 under a
    HORIZONTAL I at row 0. A real ghost would track the piece's columns;
    a hint does not, and a rule that demanded it would have refused every
    frame in both committed sessions. The piece TYPE, which test 5 does
    require, holds for both, because each is a copy of the piece in
    flight whatever is done with the rotation or the column.

    Since the correction, no committed fixture holds a game-drawn ghost
    at all, so this rule is the one part of the module argued from the
    shape space rather than from real pixels. It is kept because the SPEC
    hard requirement is game-agnosticism and most modern Tetris does draw
    a preview; it is the FALLBACK, reached only when there is no paint of
    ours to find.

    Cells the capture cannot read are left out of ALL of it — the band,
    the background cluster test 1 measures against, the neighbours tests
    3 and 4 look at, and the pieces test 5 counts as in flight (a piece
    the panel cuts in half is not a whole tetromino and names nothing).
    Their pixels are a UI panel's, and a panel is
    neither board content nor the board's background: measured on the
    live session, the covered cell (0, 8) scores 0.15 against a
    background cluster that otherwise tops out at 0.02, which is enough
    on its own to put a real ghost 0.005 inside test 1's margin and
    refuse every frame of the session.

    Two residual errors are accepted, and they are opposite corners:

    - A ghost drawn opaque enough to leave the band, or resting in a
      well, stays CONTENT. The frame then reads as it does today: a
      tetromino that cannot be explained, so the tracker holds and the
      last hint stays on screen. This is the direction the module
      prefers and the common one.
    - A real piece whose color is in the band, freshly landed on a FLAT
      surface with open air beside and above it, alone in the band,
      shaped like a tetromino AND of the same type as the piece now
      falling, is deleted. Every clause is needed, and the last one is
      what keeps it rare: the first four describe an ordinary landing on
      a pale-on-pale skin (measured over every legal placement on 4000
      random stacks — 644563 landings — 49% come to rest with nothing
      occupied to their left, right or above, so test 4 alone does NOT
      make this narrow: it makes it a coin flip), while
      test 5 additionally demands that the piece in the air be the same
      one — about one landing in seven, and none at all in the gap
      between a lock and the next spawn.
      While it lasts the cells read empty, which costs the frame rather
      than the session: the tracker sees a piece vanish, cannot explain
      it, and holds. The damage needs four such frames in a row to reach
      a BOARD_RESET, and the exposure ends the moment anything lands
      beside the piece, the band count leaves 4, or the piece in flight
      is a different one.
    """
    rows, cols = int(scores.shape[0]), int(scores.shape[1])
    observable = _observable((rows, cols), unobservable)
    band = (scores >= _GHOST_SEPARATION) & (scores < MIN_SPREAD) & observable
    if int(band.sum()) != _PIECE_CELLS:
        return None
    solid = (scores >= MIN_SPREAD) & observable
    if not bool(solid.any()):
        return None
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(band), strict=True)]
    # 1. Clear of the background cluster.
    if not _clear_of_background(scores, band, observable):
        return None
    # 2. Exactly one tetromino.
    piece = _piece_named(cells)
    if piece is None:
        return None
    # 3. Resting on the floor or on something at a real piece color.
    if not any(r + 1 >= rows or bool(solid[r + 1, c]) for r, c in cells):
        return None
    # 4. Open air beside it and above it: the stack may hold it up, but
    #    it may not have grown around it.
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and bool(solid[nr, nc]):
                return None
    # 5. A preview OF something: the piece it copies must be on the board,
    #    in flight.
    if piece not in _pieces_in_flight(solid, unobservable):
        return None
    return band


def _content_budget(
    promoted: NDArray[np.bool_],
    occupancy: NDArray[np.bool_],
    observable: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
) -> bool:
    """Is ``promoted`` an amount of board content a frame could really hold?

    The guard on the band's OCCUPIED default. That default is right, and
    the argument for it is a cost asymmetry: a candidate no rule can name
    is board content, because reading a ghost as a piece costs a
    tetromino that teleports — bad, and BOUNDED, since the tracker
    refuses to explain it and holds — while reading a piece as background
    costs the piece entirely. Both halves of that sentence are about ONE
    PIECE. At board scale it is not an asymmetry at all: a phantom the
    size of the playfield is not held, it is a new board, and four frames
    of it is a BOARD_RESET that commits the phantom as stack.

    Board scale is exactly what the band admits, because the only thing
    the band itself asks of a level is that it stand
    :data:`_GHOST_SEPARATION` clear of the background cluster — a step of
    about 14 uint8 units. A playfield drawn in TWO BACKGROUND SHADES
    clears that by construction, and those are not rare: alternating
    column or row shading, a top-out danger tint, a half-dimmed board
    behind a menu, a translucent pause panel. Measured on a 10x12 board
    against a remembered background, alternating column shading 28 units
    apart read as five phantom columns floor to ceiling at confidence
    0.31, and a danger tint over the top four rows as four phantom
    completed rows at the same number — twice the frame gate, both.

    So the promotion is bounded three ways, by the three things that make
    a reading a BOARD rather than a picture. Each catches a shading the
    other two do not, which is why all three are here:

    1. SIZE. A promoted level is being called one PIECE COLOR; seven of
       those share a board that is never full, so a level holding a
       quarter of the playfield is not a piece color, it is the playfield
       in a second shade. Measured over the five committed windows (767
       readings) the band never exceeds 8 cells of ~118 observable and
       the only promotion in any of them is 6, so this budget sits four
       times clear of every real reading. It is what catches the shadings
       that reach the FLOOR and so hang from nothing: alternating
       columns, a dimmed half-board.
    2. AIR. A board shows a settled stack plus at most one falling piece,
       so a promotion may add at most :data:`_PIECE_CELLS` cells OVER THE
       VOID — the same budget, on the same measurement, that
       :func:`_top_row_vouchable` spends on the same question. Support
       from below is the thing a floating panel cannot have. A stack band
       cut loose by its own buried holes still has its own column under
       it, so this costs a hole-riddled real stack nothing (see
       :func:`_over_the_void`). It is what catches a tint or a panel
       floating over open board, however small.
    3. COMPLETED ROWS. A completed row clears the instant it completes,
       so a promotion that finishes one describes a board the game cannot
       be showing (:func:`_claims_a_completed_row`, asked only of the
       rows the promotion actually contributes to — a row that was
       already complete without it is not this rule's business). It is
       what catches a shading drawn in FULL ROWS across a board with a
       stack under it, which is small enough for 1 and grounded enough
       for 2.

    A band that fails any of the three is not resolvable: it cannot be
    content (that is the phantom board) and it cannot be background (that
    is the pale piece lost again). It is in the same position as our own
    paint under the rotation badge, and it gets the same answer — the
    FRAME is refused, and the last committed state stands.
    """
    claimed = int(promoted.sum())
    if claimed == 0:
        return True
    if claimed >= _BAND_CONTENT_SHARE * int(observable.sum()):
        return False
    airborne, grounded = _support(occupancy, unobservable)
    if int((_over_the_void(airborne, grounded) & promoted).sum()) > _PIECE_CELLS:
        return False
    # Only the rows the promotion put something in: a row that was
    # already complete without it is not this rule's business.
    touched = occupancy & promoted.any(axis=1)[:, None]
    return not _claims_a_completed_row(touched, unobservable)


def _band_only_reading(
    colors: NDArray[np.float32],
    background: NDArray[np.float64],
    band: NDArray[np.bool_],
    unobservable: frozenset[tuple[int, int]],
    own_paint: OwnPaint | None,
) -> _Reading:
    """A board holding nothing but the background and one intermediate band.

    The uniform-empty branch's blind spot: with no cell reaching
    :data:`MIN_SPREAD` that branch calls the whole frame an empty board,
    which is exactly right for a board wipe and exactly wrong for a pale
    piece spawning onto a clear board — the moment a coach is most
    needed. Those two frames are told apart by the band standing clear of
    the background cluster, which a truly uniform frame has nothing to
    do.

    There is no two-class split here to name the layer by structure:
    :func:`_ghost_layer` needs a solid class to find the piece a preview
    would be a copy of, and finds none, which is the right answer anyway
    (a preview of nothing is not a preview). :func:`_own_paint_layer`
    still works, because it knows its color rather than the board. So
    the band is our hint if the paint says so, unresolvable if the paint
    is there but the widget was refused, and otherwise content.

    The confidence is :data:`_UNIFORM_EMPTY_CONFIDENCE`, the number this
    branch already reports, and for the reason it reports it: the
    reading is structurally grounded rather than measured from a gap.
    """
    # No cell reaches MIN_SPREAD here, so the solid class the badge test
    # consults is empty by construction.
    layer = _own_paint_layer(colors, background, np.zeros_like(band), unobservable, own_paint)
    empty = np.zeros_like(band)
    if layer is not None:
        return _Reading(empty, _UNIFORM_EMPTY_CONFIDENCE, layer)
    painted = _own_paint_cells(colors, background, unobservable, own_paint)
    if painted is not None and bool((band & painted).any()):
        return _Reading(empty, 0.0, None)
    return _Reading(band.copy(), _UNIFORM_EMPTY_CONFIDENCE, None)


def _classify_scored(
    colors: NDArray[np.float32],
    background: NDArray[np.float64],
    unobservable: frozenset[tuple[int, int]] = frozenset(),
    own_paint: OwnPaint | None = HINT_PAINT,
) -> _Reading:
    """Split per-cell colors into empty/occupied by distance from ``background``.

    Three levels, two classes. Between the background cluster and a real
    piece color there is a BAND — ``[_GHOST_SEPARATION, MIN_SPREAD)``,
    conditioned on :func:`_clear_of_background` so a continuum never
    reaches it — that a translucent layer and a pale-on-pale piece both
    land in, four thousandths apart on this game's own theme. Nothing
    about a band cell's SCORE says which it is, so the threshold is not
    allowed to decide: every band cell is a CANDIDATE, taken out of
    Otsu's hands and given to the structural rules, and what the rules
    say about it is what it becomes.

    The three dispositions, and which way each errs:

    - NAMED a landing preview by :func:`_own_paint_layer` (this tool's
      own hint, by arithmetic) or :func:`_ghost_layer` (a game-drawn
      ghost, by structure) -> EMPTY, and out of the split entirely,
      since nothing is there. Both rules err toward leaving cells alone,
      because deleting real content hands the solver room that does not
      exist.
    - Our own paint that :func:`_own_paint_layer` REFUSED to name — a
      widget the panel cut in half, or one under the rotation badge —
      cannot be called content (it is not) and cannot be deleted (the
      badge above it would be left behind as an unexplainable added
      cell). It is the one band a frame cannot resolve, so the frame
      loses its confidence instead of guessing: 0.0, below any gate.
    - Everything else -> OCCUPIED. A candidate no rule can account for
      is board content until something says otherwise, which is the
      OPPOSITE default from the one this line used to have.

    That last flip is the whole change, and the errors it trades between
    are not symmetric. Read a ghost as a piece and the board grows a
    tetromino that teleports: a phantom lock, unexplainable frames, a
    BOARD_RESET, a hint that jumps — bad, and bounded, since the tracker
    refuses to explain it and holds. Read a real piece as background and
    the piece is not there at all: no falling piece, no hint, for as
    long as it is on screen (measured on ``tests/fixtures/pale_piece``:
    61 frames of a pale periwinkle T absent from the occupancy, and 90
    frames — ~6 s — with nothing on the overlay). The structural rules
    still get first refusal, so a ghost they can name is still deleted;
    what changes is where the silence falls, and it now falls on the
    side of a held frame rather than a lost piece.
    """
    rows, cols = int(colors.shape[0]), int(colors.shape[1])
    scores = _distance_scores(colors, background)
    lo = float(scores.min())
    hi = float(scores.max())
    observable = _observable((rows, cols), unobservable)
    band = (scores >= _GHOST_SEPARATION) & (scores < MIN_SPREAD) & observable
    if not _clear_of_background(scores, band, observable):
        # Not a level of its own: the background's own noise, or a
        # continuum with no gap anywhere in it. Reads as it always did.
        band = np.zeros((rows, cols), dtype=np.bool_)
    if hi < MIN_SPREAD and bool(band.any()):
        # A band standing clear of the background, and nothing on the
        # board reaching the floor for "a different color": a pale piece
        # on an otherwise empty board — the failure this function exists
        # to fix, at the one moment there is no two-class split to read
        # it against. Structure is all there is, so the reading is
        # entirely structural and says so with the same number a
        # uniform-empty frame reports.
        return _band_only_reading(colors, background, band, unobservable, own_paint)
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
        return _Reading(np.zeros((rows, cols), dtype=np.bool_), _UNIFORM_EMPTY_CONFIDENCE, None)
    if hi - lo < MIN_SPREAD:
        # Uniform-far: every cell is far from the background estimate yet
        # mutually similar — the estimator is inconsistent with the field
        # (a heterogeneous banner drawn across the top row over a solid
        # field; a solid pause panel or flash in a color that is NOT a
        # remembered background). Truly unreadable: described as
        # all-occupied at zero confidence so the caller's gate rejects it;
        # it is never classified as empty.
        return _Reading(np.ones((rows, cols), dtype=np.bool_), 0.0, None)

    # A landing preview is a third level between the two classes; name it
    # and take it out of the split entirely, so the threshold and the
    # confidence below are both measured on the separation that decides
    # occupancy — background versus a real piece color. Left in, it drags
    # whichever class Otsu attaches it to toward the other one, which is
    # how a cleanly readable frame ends up reported as ambiguous (measured
    # on the fifteen live_session frames that carry one: 0.078 with the
    # layer left in the empty class, rejected at a 0.15 gate, against
    # 0.286 with it named). What the frame may then report is capped: see
    # :data:`_LAYER_CONFIDENCE_CEILING`.
    # Two rules name that layer, asked in order of what they know. The
    # coach's OWN hint overlay is in its own capture and its color is
    # known in advance, so _own_paint_layer decides it by measurement;
    # only when there is no paint to find does _ghost_layer's structural
    # rule get a turn, for the games that really do draw a landing
    # preview. They are alternatives rather than a union on purpose: a
    # frame carrying BOTH our paint and a real ghost has the paint taken
    # out and the ghost left in, which is a held frame — the direction
    # every rule here errs in. No committed fixture contains one (see
    # _own_paint_layer: this game draws no ghost at all), so a
    # composition would be written against nothing.
    ghost = _own_paint_layer(
        colors, background, (scores >= MIN_SPREAD) & observable, unobservable, own_paint
    )
    if ghost is None:
        ghost = _ghost_layer(scores, unobservable)
    board = np.ones((rows, cols), dtype=np.bool_) if ghost is None else ~ghost

    # Whatever the rules did NOT name is still in the band, and the
    # threshold does not get to decide it either. Our own paint is the
    # one thing that can be neither deleted nor believed; everything
    # else is content (see this function's own docstring).
    candidates = band & board
    painted = _own_paint_cells(colors, background, unobservable, own_paint)
    unresolved = candidates & painted if painted is not None else np.zeros_like(candidates)
    content = candidates & ~unresolved

    threshold = otsu_threshold(scores[board])
    occupancy = (scores > threshold) & board
    empty_scores = scores[board & ~occupancy]
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
        occupancy = (scores > MIN_SPREAD) & board
        empty_scores = scores[board & ~occupancy]
    promoted = content & ~occupancy
    if bool(promoted.any()):
        # A candidate the rules could not name is board content, so it
        # joins the occupied class — including for the gap below, which
        # then reports the separation that actually decided this frame:
        # the band against the background, rather than the solid pieces
        # against a band silently filed with them.
        #
        # Only the cells the threshold DROPPED are promoted, which is the
        # whole of the change on real frames: where Otsu had already put
        # the band with the pieces (``tests/fixtures/absorbed_piece``, and
        # the settled periwinkle of ``tests/fixtures/roas_stacker``) this
        # is a no-op and those readings are byte-identical to what they
        # always were, confidence included.
        occupancy = occupancy | promoted
        empty_scores = scores[board & ~occupancy]
        if not _content_budget(promoted, occupancy, observable, unobservable):
            # More content than a board can hold. The band's own
            # admission test is a ~14 uint8 step, which a playfield drawn
            # in two background shades clears by construction, so this
            # default would otherwise turn alternating column shading or
            # a top-out danger tint into a full board of phantom stack
            # at twice the gate. Unresolvable, and refused for the same
            # reason our own unnameable paint is.
            return _Reading(occupancy, 0.0, ghost)
    occupied_scores = scores[occupancy]
    if occupied_scores.size == 0 or empty_scores.size == 0:
        return _Reading(occupancy, 0.0, ghost)
    if bool(unresolved.any()):
        # Our own paint, in the band, that _own_paint_layer refused to
        # name. Deleting it strands the rotation badge as an added cell
        # nothing can explain; calling it content invents a tetromino
        # where the coach's own overlay is. Neither, then: the frame is
        # refused outright. (Measured on live_session 121-135, the 14
        # badge frames: 0.078 before, which the 0.15 gate rejected by
        # four hundredths of an accident. This says it on purpose.)
        return _Reading(occupancy, 0.0, ghost)
    # Fraction of the observed score range separating the two classes:
    # 1.0 when the split is wide open, near 0 when samples nearly touch.
    # The range stays the FULL one: a ghost lies strictly inside it, so
    # dropping the layer changes the gap, never the scale it is read on.
    gap = float(occupied_scores.min() - empty_scores.max())
    if bool(promoted.any()):
        # A promoted band is a LEVEL being called content over the
        # threshold's objection, and a level needs air on BOTH sides. The
        # gap above already reports the air below it (the band is now the
        # lowest occupied thing there is); this is the other side, and it
        # is what refuses a lighting gradient, whose band cells are one
        # slice of a ramp with the next slice 0.074 above them. No new
        # threshold decides it: the weakest boundary the three-level
        # reading rests on IS the frame's confidence, and at the gate a
        # ramp falls (measured: 0.103 on a paper-white board under a
        # 140-unit vertical gradient, against 0.206 for the real pale
        # piece of tests/fixtures/pale_piece, whose next level up is
        # 0.469 away).
        #
        # The air is looked for on the BOARD, not on the frame: a named
        # layer's cells were just declared not to be there, so they
        # cannot be the next level up either. Left in, a theme whose
        # paint composites just over MIN_SPREAD (black ground: 0.374,
        # see _own_paint_layer) puts a hint 0.028 above a promoted pale
        # piece and collapses a correct reading from 0.433 to 0.035,
        # under the gate — the pale-piece failure back again, now by way
        # of the confidence rather than the split.
        gap = min(gap, _clearance_above(scores, promoted, observable & board))
    confidence = float(np.clip(gap / (hi - lo), 0.0, 1.0))
    if ghost is not None:
        # Three levels were seen and two are being reported on; the third
        # is accounted for by structure, not by this number (see
        # :data:`_LAYER_CONFIDENCE_CEILING`).
        confidence = min(confidence, _LAYER_CONFIDENCE_CEILING)
    return _Reading(occupancy, confidence, ghost)


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
        own_paint: OwnPaint | None = HINT_PAINT,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._min_confidence = min_confidence
        self._unobservable: frozenset[tuple[int, int]] = unobservable_cells or frozenset()
        self._own_paint = own_paint
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
            reading = _classify_scored(
                colors, self._background, self._unobservable, self._own_paint
            )
            if reading.confidence >= self._min_confidence:
                if self._confirmed and not self._corroborated:
                    agrees = self._second_opinion(colors)
                    if agrees is False:
                        # An anchor and a from-scratch reading of the same
                        # frame that disagree about the background cannot
                        # both be the board: believe neither.
                        self._forget()
                        return reading.occupancy, 0.0
                    self._corroborated = agrees is True
                if _claims_a_completed_row(reading.occupancy, self._unobservable):
                    # The anchor is describing a board that cannot exist.
                    return reading.occupancy, self._impossible_frame(colors)
                self._remember(colors, reading)
                return reading.occupancy, reading.confidence
            if self._confirmed:
                return reading.occupancy, reading.confidence
            # Provisional memory failed on this frame: fall through and
            # re-bootstrap, so a wrong provisional anchor cannot wedge
            # the session before play has even been observed.
        reading = self._bootstrap(colors)
        if reading.confidence >= self._min_confidence:
            self._remember(colors, reading)
        return reading.occupancy, reading.confidence

    def _bootstrap(self, colors: NDArray[np.float32]) -> _Reading:
        """Self-estimated classification: classify_grid's memoryless path."""
        return _self_estimated(colors, self._unobservable, self._own_paint)

    def _remember(self, colors: NDArray[np.float32], reading: _Reading) -> None:
        """Re-measure the background from an accepted frame.

        Unobservable cells are left out of every sample: their pixels are
        the covering panel's, and a memory measured partly from the panel
        would drift toward a color the board never shows. So are the cells
        of a landing preview (:attr:`_Reading.ghost`): they read EMPTY,
        and nothing is there, but their color is a piece blended into the
        board, so averaging them into the anchor would walk it toward the
        pieces — the one direction that inverts a reading.

        Reaching here is also what ends a run of impossible frames: this
        is called exactly when a frame was accepted AND describes a board
        that can exist, on the bootstrap path as well as the remembered
        one, so no count survives into the next anchor.
        """
        self._impossible = 0
        occupancy = reading.occupancy
        observable = self._observable_mask(occupancy.shape)
        if reading.ghost is not None:
            observable = observable & ~reading.ghost
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
        reading = self._bootstrap(colors)
        if reading.confidence < self._min_confidence:
            return None
        occupancy = reading.occupancy
        observable = self._observable_mask(occupancy.shape)
        if reading.ghost is not None:
            observable = observable & ~reading.ghost
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

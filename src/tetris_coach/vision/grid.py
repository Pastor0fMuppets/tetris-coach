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
200 cell scores are then split by Otsu's method into empty/occupied
classes; distance-from-background makes the polarity fixed by
construction (high score = occupied) — provided the estimate really is
the background, which is exactly what the top-row prior cannot guarantee
when a stack legally reaches the visible top row (see
:func:`classify_grid`'s strict cap).

Channel order does not matter (BGR vs RGB): Euclidean distance and the
per-channel median are permutation-equivariant in the channel axis.
A 2-D grayscale image is simply the single-channel (C=1) case.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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


def cell_scores(
    image: NDArray[np.uint8],
    rows: int = 20,
    cols: int = 10,
    margin: float = 0.25,
) -> NDArray[np.float32]:
    """Per-cell occupancy score in [0, 1]: distance from the background.

    The background is estimated per frame as the per-channel median of the
    TOP ROW's cell colors — a gravity prior: the top row is usually
    background, contaminated by at most the 4 cells of a freshly spawned
    piece, so the median is taken over a majority of true background
    cells. This recovers even boards that are mostly filled (where any
    dominant-cluster estimate would lock onto the pieces and invert the
    reading).

    The prior's limit: a STACK reaching the visible top row is legal,
    reachable Tetris (side columns stacked to row 0 while the spawn
    columns stay clear; versus-mode garbage rows, 9/10 filled, pushed up
    to the top) — it is NOT game over. With >= 6 of the 10 top-row cells
    non-background the median locks onto the pieces and the polarity of
    every score inverts. :func:`classify_grid` therefore refuses to vouch
    for a reading whose own top row comes out occupied (strict cap);
    streaming callers should use :class:`GridClassifier`, whose committed
    background memory keeps such boards readable at full confidence.
    """
    colors = _cell_colors(image, rows, cols, margin)
    background = np.median(colors[0], axis=0)
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
    exact path for small value sets (e.g. the 200 cell scores of a board).
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
    rows: int = 20,
    cols: int = 10,
    background: NDArray[np.float64] | tuple[float, ...] | None = None,
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

    Without ``background`` the top-row-median estimate is used, under one
    strict cap: a two-class reading whose OWN top row contains occupied
    cells contradicts the estimate's majority-background prior — the very
    configuration in which the median can lock onto piece colors and
    invert every cell at high apparent confidence (a legal, reachable
    state: side columns stacked to row 0, versus garbage pushed to the
    top; see :func:`cell_scores`). Such readings keep their occupancy
    (still exact in the benign spawned-piece case) but are capped to
    confidence 0.0: without cross-frame memory their polarity cannot be
    vouched for, and a wrong reading above the gate is worse than a
    dropped frame.
    """
    colors = _cell_colors(image, rows, cols, margin=0.25)
    if background is not None:
        return _classify_scored(colors, np.asarray(background, dtype=np.float64))
    occupancy, confidence = _classify_scored(colors, np.median(colors[0], axis=0))
    if confidence > 0.0 and bool(occupancy[0].any()):
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
      prior, strict cap included. An accepted two-class frame anchors the
      memory on its empty-class median and CONFIRMS it; an accepted
      uniform-empty frame anchors it provisionally (the frame's overall
      median) — a solid frame alone cannot prove it is the board.
    - Provisional memory: used first; if the frame is rejected under it,
      the frame re-bootstraps (a wrong provisional anchor, e.g. a pause
      panel covering the board at startup, must not blind the session).
      The first accepted two-class frame confirms the memory.
    - Confirmed memory: never falls back and never re-anchors — a frame
      it cannot read is rejected, holding the last committed state. This
      is deliberate: the plausible-looking alternatives (a top-row
      re-estimate on a garbage board, a bright overlay re-read as a
      light-theme empty board) are exactly the wrong readings the memory
      exists to prevent. It still tracks slow drift, re-measuring from
      every accepted frame.

    Single-threaded use only (the engine's worker thread).
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 10,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._min_confidence = min_confidence
        self._background: NDArray[np.float64] | None = None
        self._confirmed = False

    @property
    def background(self) -> NDArray[np.float64] | None:
        """Remembered background color; None before any anchor."""
        return None if self._background is None else self._background.copy()

    @property
    def confirmed(self) -> bool:
        """True once an accepted two-class frame anchored the memory."""
        return self._confirmed

    def classify(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.bool_], float]:
        """Classify one frame; same return contract as :func:`classify_grid`."""
        colors = _cell_colors(image, self._rows, self._cols, margin=0.25)
        if self._background is not None:
            occupancy, confidence = _classify_scored(colors, self._background)
            if confidence >= self._min_confidence:
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
        occupancy, confidence = _classify_scored(colors, np.median(colors[0], axis=0))
        if confidence > 0.0 and bool(occupancy[0].any()):
            confidence = 0.0  # the strict cap, verbatim (see classify_grid)
        return occupancy, confidence

    def _remember(self, colors: NDArray[np.float32], occupancy: NDArray[np.bool_]) -> None:
        """Re-measure the background from an accepted frame."""
        flat = colors.reshape(-1, colors.shape[-1])
        if bool(occupancy.any()):
            # Two-class frame: the empty class IS the background, freshly
            # measured. Anchoring here is what makes polarity survive the
            # stack growing into row 0.
            self._background = np.asarray(
                np.median(flat[~occupancy.ravel()], axis=0), dtype=np.float64
            )
            self._confirmed = True
        else:
            # Uniform-empty frame: the whole frame is the background —
            # but only provisionally, since a solid pause panel is
            # indistinguishable from an empty board (never CONFIRM from
            # one, or a panel at startup would poison the session).
            self._background = np.asarray(np.median(flat, axis=0), dtype=np.float64)

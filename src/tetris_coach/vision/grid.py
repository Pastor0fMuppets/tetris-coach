"""Occupancy classification of a board-region image.

Game-agnostic: works on *occupancy*, not colors — and not on any absolute
brightness scale either. Each cell of the (rows x cols) grid is sampled at
the central portion of its patch (skipping gridlines) and scored by its
color's Euclidean distance from a per-frame estimate of the board's own
background color, so 0 means "at the background" BY CONSTRUCTION whatever
the theme: dark boards with bright pieces, white boards with colored
pieces, and colored backgrounds all score the same way. The darker-
background restriction of earlier versions is lifted at this layer.

The background estimate is the per-channel median of the TOP ROW's cell
colors (see :func:`cell_scores` for the gravity-prior argument). The 200
cell scores are then split by Otsu's method into empty/occupied classes;
distance-from-background makes the polarity fixed by construction (high
score = occupied).

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
# two uniform-region rules cannot drift apart.
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
    TOP ROW's cell colors — a gravity prior: on a legal board a spawning
    piece covers at most 4 of the top row's 10 cells (so the median is
    taken over a majority of true background cells), and a *stack* reaching
    row 0 is game over. This recovers even boards that are mostly filled
    (where any dominant-cluster estimate would lock onto the pieces and
    invert the reading).
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
) -> tuple[NDArray[np.bool_], float]:
    """Classify a board-region image into per-cell occupancy.

    Returns ``(occupancy, confidence)`` where ``occupancy`` is a
    (rows, cols) bool array (True = occupied) and ``confidence`` in [0, 1]
    reflects how cleanly the two classes separate. Confidence semantics:
    0.0 = unreadable, the caller's gate must reject; 0.5 = trusted
    uniform-empty; (0, 1] = class-gap fraction for a two-class frame.
    """
    scores = cell_scores(image, rows, cols)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi < MIN_SPREAD:
        # Uniform-near: every cell sits at the frame's own background
        # estimate — an EMPTY board, whatever color the theme paints it
        # (a solid near-white region is a light theme's empty board just
        # as a solid dark one is a dark theme's). Classified empty with
        # usable confidence so a board wipe (game over, new game) reaches
        # the tracker instead of being dropped at the gate. A whole-board
        # flash or a solid pause overlay reads the same way — the frames
        # are pixel-indistinguishable from an empty board — and the
        # tracker's reset debounce (several consecutive identical
        # unexplained frames) is what keeps short flashes from committing
        # anything; that is where the cross-frame information lives.
        return np.zeros((rows, cols), dtype=np.bool_), _UNIFORM_EMPTY_CONFIDENCE
    if hi - lo < MIN_SPREAD:
        # Uniform-far: every cell is far from the top-row background
        # estimate yet mutually similar — the estimator is inconsistent
        # with the field (e.g. a heterogeneous banner drawn across the top
        # row over a solid field). Truly unreadable: described as
        # all-occupied at zero confidence so the caller's gate rejects it;
        # it is never classified as empty.
        return np.ones((rows, cols), dtype=np.bool_), 0.0

    threshold = otsu_threshold(scores)
    occupancy = scores > threshold

    occupied_scores = scores[occupancy]
    empty_scores = scores[~occupancy]
    if occupied_scores.size == 0 or empty_scores.size == 0:
        return occupancy, 0.0
    # Fraction of the observed score range separating the two classes:
    # 1.0 when the split is wide open, near 0 when samples nearly touch.
    gap = float(occupied_scores.min() - empty_scores.max())
    confidence = float(np.clip(gap / (hi - lo), 0.0, 1.0))
    return occupancy, confidence

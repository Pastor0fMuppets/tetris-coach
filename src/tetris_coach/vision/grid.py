"""Occupancy classification of a board-region image.

Game-agnostic: works on *occupancy*, not colors. Each cell of the (rows x
cols) grid is scored by sampling the central portion of its patch (skipping
gridlines) and combining brightness and saturation — occupied cells are
either bright or strongly colored, while the background is darker and duller
(a hard requirement of the product). The 200 cell scores are then split by
Otsu's method into empty/occupied classes.

Channel order does not matter (BGR vs RGB): brightness is the per-pixel
channel maximum and saturation the normalized max-min spread, both of which
are permutation-invariant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Minimum score spread before Otsu thresholding is considered meaningful;
# below this the region is treated as uniform (an empty board).
_MIN_SPREAD = 0.15


def _build_score_lut() -> NDArray[np.float32]:
    """256x256 lookup of the color score for a (channel max, channel min) pair.

    The score of a color pixel depends only on its uint8 channel max and
    min, so all 65536 combinations are precomputed once — with exactly the
    float32 arithmetic documented in :func:`score_map`, making table lookups
    bit-identical to computing the formula per pixel.
    """
    mx = (np.arange(256).astype(np.float32) / 255.0)[:, None]
    mn = (np.arange(256).astype(np.float32) / 255.0)[None, :]
    # HSV-style saturation, but with the denominator clamped so that sensor
    # noise on near-black pixels cannot masquerade as strong color.
    saturation = (mx - mn) / np.maximum(mx, 0.25)
    return np.maximum(mx, saturation).astype(np.float32)


_SCORE_LUT = _build_score_lut()


def score_map(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Per-pixel occupancy score in [0, 1]: max(brightness, saturation).

    Brightness is the channel max ``mx`` (as a 0..1 float) and saturation
    the clamped HSV-style spread ``(mx - mn) / max(mx, 0.25)``. The channel
    max/min are reduced on the raw uint8 pixels (``x -> x/255`` is monotone,
    so the reduction commutes with the conversion) and the score comes from
    a precomputed 256x256 table: one float lane of work per pixel instead
    of five, with bit-identical results.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
    # Elementwise plane max/min: identical to ``.max(axis=2)`` but ~4x
    # faster (numpy's 3-element axis reduce pays iterator overhead per
    # pixel; the two-plane ufunc runs a straight vector loop).
    c0, c1, c2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(c0, c1), c2)
    mn = np.minimum(np.minimum(c0, c1), c2)
    return _SCORE_LUT[mx, mn]


def cell_scores(
    image: NDArray[np.uint8],
    rows: int = 20,
    cols: int = 10,
    margin: float = 0.25,
) -> NDArray[np.float32]:
    """Mean occupancy score of the central patch of every cell.

    ``margin`` is the fraction of each cell inset on every side before
    sampling, which skips gridlines and cell borders.

    Only the sampled sub-rects are scored: with the default margin that is
    a quarter of the image, so running :func:`score_map` per patch beats
    scoring the whole image up front by ~4x with identical results.
    """
    img = np.asarray(image)
    h, w = int(img.shape[0]), int(img.shape[1])
    if h < rows or w < cols:
        raise ValueError(f"image {w}x{h} too small for a {cols}x{rows} grid")
    ys = np.linspace(0, h, rows + 1)
    xs = np.linspace(0, w, cols + 1)
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        cell_h = float(ys[r + 1] - ys[r])
        y0 = round(float(ys[r]) + cell_h * margin)
        y1 = max(y0 + 1, round(float(ys[r + 1]) - cell_h * margin))
        for c in range(cols):
            cell_w = float(xs[c + 1] - xs[c])
            x0 = round(float(xs[c]) + cell_w * margin)
            x1 = max(x0 + 1, round(float(xs[c + 1]) - cell_w * margin))
            out[r, c] = score_map(img[y0:y1, x0:x1]).mean()
    return out


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
    reflects how cleanly the two classes separate.
    """
    scores = cell_scores(image, rows, cols)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < _MIN_SPREAD:
        # Uniform region: no pieces distinguishable from background. Given
        # the darker-background requirement, treat it as an empty board.
        occupancy = np.zeros((rows, cols), dtype=np.bool_)
        return occupancy, 0.5

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

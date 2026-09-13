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


def score_map(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Per-pixel occupancy score in [0, 1]: max(brightness, saturation)."""
    img = np.asarray(image)
    if img.ndim == 2:
        return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
    channels = img[:, :, :3].astype(np.float32) / 255.0
    mx = channels.max(axis=2)
    mn = channels.min(axis=2)
    # HSV-style saturation, but with the denominator clamped so that sensor
    # noise on near-black pixels cannot masquerade as strong color.
    saturation = (mx - mn) / np.maximum(mx, 0.25)
    return np.maximum(mx, saturation).astype(np.float32)


def cell_scores(
    image: NDArray[np.uint8],
    rows: int = 20,
    cols: int = 10,
    margin: float = 0.25,
) -> NDArray[np.float32]:
    """Mean occupancy score of the central patch of every cell.

    ``margin`` is the fraction of each cell inset on every side before
    sampling, which skips gridlines and cell borders.
    """
    scores = score_map(image)
    h, w = scores.shape
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
            out[r, c] = scores[y0:y1, x0:x1].mean()
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

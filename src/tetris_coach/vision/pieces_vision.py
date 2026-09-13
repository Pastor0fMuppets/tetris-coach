"""Shape recognition: falling piece in the board grid, next piece in a preview image.

Both work purely on shape (occupancy); color is never required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.pieces import ROTATIONS
from .grid import otsu_threshold, score_map

Cell = tuple[int, int]

# Normalized cell set -> (piece, rotation index), for O(1) shape matching.
_SHAPE_LOOKUP: dict[tuple[Cell, ...], tuple[str, int]] = {
    rot.cells: (rot.piece, rot.index)
    for rots in ROTATIONS.values()
    for rot in rots
}


@dataclass(frozen=True)
class FallingPiece:
    """A recognized falling piece, positioned on the board grid."""

    piece: str
    rotation_index: int
    row: int  # top row of the bounding box on the board
    col: int  # left column of the bounding box on the board


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


def _connected_components(grid: NDArray[np.bool_]) -> list[list[Cell]]:
    """4-connected components of occupied cells."""
    rows, cols = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    components: list[list[Cell]] = []
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            component: list[Cell] = []
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(component)
    return components


def split_grid(
    grid: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], FallingPiece | None]:
    """Separate the stack from the falling piece in a full occupancy grid.

    The falling piece is a 4-cell connected component that does not touch
    the bottom row and matches a tetromino shape; everything else (including
    floating debris left by clear animations) is treated as stack. When
    several components qualify, the topmost is taken as the falling piece.
    """
    rows = grid.shape[0]
    components = _connected_components(grid)

    candidates: list[tuple[int, list[Cell], tuple[str, int]]] = []
    for component in components:
        if len(component) != 4:
            continue
        if any(r == rows - 1 for r, _ in component):
            continue  # touches the floor: part of the stack
        matched = match_cells(tuple(component))
        if matched is None:
            continue
        top = min(r for r, _ in component)
        candidates.append((top, component, matched))

    if not candidates:
        return grid.copy(), None

    candidates.sort(key=lambda item: item[0])
    top, component, (piece, rotation_index) = candidates[0]
    stack = grid.copy()
    for r, c in component:
        stack[r, c] = False
    falling = FallingPiece(
        piece=piece,
        rotation_index=rotation_index,
        row=top,
        col=min(c for _, c in component),
    )
    return stack, falling


def identify_falling(grid: NDArray[np.bool_]) -> FallingPiece | None:
    """Recognize the falling piece in a full occupancy grid (or ``None``)."""
    _, falling = split_grid(grid)
    return falling


def identify_next(image: NDArray[np.uint8]) -> str | None:
    """Recognize the piece shown in a next-piece preview image.

    Thresholds the image, crops the foreground to its bounding box,
    resamples it to each candidate rotation's cell grid, and returns the
    piece whose shape signature matches exactly (or ``None``).
    """
    scores = score_map(image)
    flat = scores.ravel()
    if float(flat.max()) - float(flat.min()) < 0.15:
        return None  # empty preview box
    mask = scores > otsu_threshold(flat.astype(np.float32))

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
            occupancy = _resample_to_cells(crop, rot.height, rot.width)
            if occupancy is None:
                continue
            expected = np.zeros((rot.height, rot.width), dtype=bool)
            for r, c in rot.cells:
                expected[r, c] = True
            if not np.array_equal(occupancy[0], expected):
                continue
            fit = occupancy[1]
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

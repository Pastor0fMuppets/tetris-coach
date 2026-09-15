"""Reading the NEXT box: which tetromino it holds, and in what colour.

The NEXT reader the default tracker uses (see :mod:`.colour_palette`); the
other one is ``pieces_vision.identify_next``, which ``--tracker shape``
still reads with. This one answers a second question that one cannot, and
the tracker needs: the piece's COLOUR. That is the
labelled example the whole colour-first design runs on — the box says "this
colour is a T" for free, every time a piece is dealt, with no shape
ambiguity to resolve and no ghost in the way.

The reading is deliberately blunt: the box holds one flat-coloured piece on
a plain ground, so the piece is the largest blob of the box's most common
non-background, non-our-paint colour, and the piece is named by pooling that
blob into each candidate rotation's cell grid. Aspect ratio picks the
candidate shapes (a 2x2 O cannot be read as a 1x4 I), coverage decides
between same-shaped ones (T, S, Z, J and L are all 2x3), and a coverage
floor keeps the box's own furniture out: this tool's own rotation badge is a
disc, which fills 0.785 of its cell and cannot reach :data:`MIN_COVERAGE`,
and its translucent fill is refused outright by :data:`PAINT_FILL_MAX`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ..core.pieces import ROTATIONS
from .colour_palette import EMPTY_DIST, LINE_TOL
from .grid import HINT_PAINT, OwnPaint

# Fraction of a rotation's own cells the blob must cover, and of its empty
# cells it may spill into, for the reading to be that rotation.
MIN_COVERAGE = 0.85
MAX_SPILL = 0.30

# Tolerated relative error between the blob's aspect ratio and a rotation's.
ASPECT_TOL = 0.25

# Fraction of the hint colour's own magnitude below which a pixel lying on
# the hint's ray away from the box background is this tool's translucent
# fill rather than a piece that happens to be that colour. The box is read
# at pixel level, where the fill has no outline of its own to be signed by
# (see colour_palette.own_paint_states, which does have cells to work with).
PAINT_FILL_MAX = 0.8

# Smallest blob taken for a piece cell, in pixels, and the share of the
# biggest blob a blob must reach to count as part of the same piece.
MIN_BLOB_AREA = 32
MIN_BLOB_SHARE = 0.3

_SHAPES: list[tuple[str, NDArray[np.bool_]]] = []
for _piece, _rotations in ROTATIONS.items():
    for _rotation in _rotations:
        _grid = np.zeros((_rotation.height, _rotation.width), dtype=bool)
        for _r, _c in _rotation.cells:
            _grid[_r, _c] = True
        _SHAPES.append((_piece, _grid))


@dataclass(frozen=True)
class PreviewReading:
    """What the NEXT box holds this frame."""

    piece: str
    colour: NDArray[np.float64]  # the piece's rendered BGR in the box
    background: NDArray[np.float64]  # the box's own ground


def _pool(mask: NDArray[np.bool_], rows: int, cols: int) -> NDArray[np.float64]:
    """Mean coverage of ``mask`` over a ``rows`` x ``cols`` grid of blocks."""
    ys = np.linspace(0, mask.shape[0], rows + 1).round().astype(int)
    xs = np.linspace(0, mask.shape[1], cols + 1).round().astype(int)
    out = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = mask[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            out[r, c] = float(block.mean()) if block.size else 0.0
    return out


def _name_blob(mask: NDArray[np.bool_]) -> str | None:
    """The tetromino this blob's bounding box spells, or ``None``."""
    aspect = mask.shape[1] / mask.shape[0]
    best: tuple[float, str] | None = None
    for piece, grid in _SHAPES:
        rows, cols = grid.shape
        if abs(aspect / (cols / rows) - 1.0) > ASPECT_TOL:
            continue
        pooled = _pool(mask, rows, cols)
        covered = float(pooled[grid].min())
        spilled = float(pooled[~grid].max()) if (~grid).any() else 0.0
        if covered < MIN_COVERAGE or spilled > MAX_SPILL:
            continue
        score = covered - spilled
        if best is None or score > best[0]:
            best = (score, piece)
    return None if best is None else best[1]


def _is_own_paint(
    vectors: NDArray[np.float64], background: NDArray[np.float64], paint: OwnPaint | None
) -> NDArray[np.bool_]:
    """Per-pixel mask of this tool's own hint paint over the box."""
    if paint is None:
        return np.zeros(vectors.shape[0], dtype=bool)
    colour = np.asarray(paint.color, dtype=np.float64)
    if colour.shape != background.shape:
        return np.zeros(vectors.shape[0], dtype=bool)
    full = colour - background
    length = float(np.linalg.norm(full))
    if length < EMPTY_DIST:
        return np.zeros(vectors.shape[0], dtype=bool)
    unit = full / length
    along = vectors @ unit
    perpendicular = np.sqrt(np.maximum(0.0, (vectors * vectors).sum(axis=1) - along * along))
    magnitude = np.linalg.norm(vectors, axis=1)
    return (perpendicular <= LINE_TOL) & (magnitude < PAINT_FILL_MAX * length)


def identify_preview(
    crop: NDArray[np.uint8], paint: OwnPaint | None = HINT_PAINT
) -> PreviewReading | None:
    """Name and colour the tetromino in a NEXT-box crop, or ``None``."""
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        return None
    pixels = crop.reshape(-1, 3).astype(np.float64)
    background = np.median(pixels, axis=0)
    vectors = pixels - background
    content = (np.linalg.norm(vectors, axis=1) > EMPTY_DIST) & ~_is_own_paint(
        vectors, background, paint
    )
    if int(content.sum()) < 32:
        return None
    # The piece is flat-coloured, so its exact value is the mode of what is
    # left; anti-aliased edges and thin furniture cannot outvote a tetromino.
    values, counts = np.unique(pixels[content].astype(np.int64), axis=0, return_counts=True)
    colour = values[int(np.argmax(counts))].astype(np.float64)
    blob = (np.linalg.norm(pixels - colour, axis=1) <= LINE_TOL).reshape(crop.shape[:2])
    count, labelled, stats, _ = cv2.connectedComponentsWithStats(
        blob.astype(np.uint8), connectivity=8
    )
    if count < 2:
        return None
    # A piece is not one blob: this game draws each of its cells as its own
    # rounded square with a gap between them. Keep every blob within
    # MIN_BLOB_SHARE of the biggest — the piece's own cells — and drop the
    # rest, which is how the anti-aliased speck in a box corner (1-2 px, and
    # exactly the piece's colour) stops stretching the bounding box over the
    # whole crop.
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = float(areas.max())
    if biggest < MIN_BLOB_AREA:
        return None
    keep = 1 + np.nonzero(areas >= MIN_BLOB_SHARE * biggest)[0]
    piece_mask = np.isin(labelled, keep)
    ys, xs = np.nonzero(piece_mask)
    piece = _name_blob(piece_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    if piece is None:
        return None
    return PreviewReading(piece=piece, colour=colour, background=background)

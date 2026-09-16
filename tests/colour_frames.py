"""Synthetic frames for the colour-first tracker's tests.

Every frame these tests use is drawn here from a grid of flat colours, so
each test states exactly the rendering it is about. The colours are the ones
measured off the real fixtures (``tests/fixtures/*``), which is what keeps a
synthetic case honest — and the hint is composited the way
``overlay.renderer.draw_hint`` composites it, fill inside a solid outline,
because recognizing that signature is what keeps the coach's own paint out
of its own reading.
"""

from __future__ import annotations

import numpy as np

from tetris_coach.overlay.renderer import rotation_badge_rect
from tetris_coach.solver.search import Move
from tetris_coach.vision.colour_tracker import ColourTracker
from tetris_coach.vision.grid import HINT_FILL_OPACITY, HINT_PAINT

ROWS, COLS, CELL = 12, 10, 20

WHITE = (252.0, 252.0, 251.0)  # the fixtures' board background, BGR
BLUE_I = (215.0, 46.0, 45.0)
GREEN_O = (112.0, 239.0, 217.0)
PALE_T = (251.0, 224.0, 206.0)
HINT = tuple(HINT_PAINT.color) if HINT_PAINT else (255.0, 229.0, 0.0)


def render(
    cells: dict[tuple[int, int], tuple[float, ...]],
    rows: int = ROWS,
    cols: int = COLS,
    background: tuple[float, ...] = WHITE,
    cell: int = CELL,
) -> np.ndarray:
    """A board image with ``cells`` painted in flat colour."""
    image = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    image[:, :] = np.array(background, dtype=np.uint8)
    for (r, c), colour in cells.items():
        image[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = np.array(
            colour, dtype=np.uint8
        )
    return image


def paint_hint(image: np.ndarray, cells: list[tuple[int, int]], cell: int = CELL) -> np.ndarray:
    """Draw this tool's own hint over ``cells``: fill inside a solid outline."""
    out = image.copy()
    hint = np.array(HINT, dtype=np.float64)
    for r, c in cells:
        y0, y1, x0, x1 = r * cell, (r + 1) * cell, c * cell, (c + 1) * cell
        under = out[y0:y1, x0:x1].astype(np.float64)
        out[y0:y1, x0:x1] = (under + HINT_FILL_OPACITY * (hint - under)).round().astype(np.uint8)
        out[y0 : y0 + 3, x0:x1] = hint
        out[y1 - 3 : y1, x0:x1] = hint
        out[y0:y1, x0 : x0 + 3] = hint
        out[y0:y1, x1 - 3 : x1] = hint
    return out


def blend(
    colour: tuple[float, ...], alpha: float, over: tuple[float, ...] = WHITE
) -> tuple[float, ...]:
    """``colour`` drawn at ``alpha`` opacity over ``over``."""
    return tuple(o + alpha * (v - o) for v, o in zip(colour, over))


def preview(
    piece_cells: list[tuple[int, int]], colour: tuple[float, ...], cell: int = CELL
) -> np.ndarray:
    """A NEXT-box crop: the piece drawn small and off-centre, as games do."""
    box = np.zeros((5 * cell, 5 * cell, 3), dtype=np.uint8)
    box[:, :] = np.array(WHITE, dtype=np.uint8)
    for r, c in piece_cells:
        y, x = (r + 1) * cell, (c + 1) * cell
        box[y : y + cell - 1, x : x + cell - 1] = np.array(colour, dtype=np.uint8)
    return box


def tracker(**kwargs: object) -> ColourTracker:
    return ColourTracker(rows=ROWS, cols=COLS, **kwargs)  # type: ignore[arg-type]


def paint_badge(image: np.ndarray, move: Move, cell: int = CELL) -> np.ndarray:
    """Draw the hint's rotation badge where the renderer draws it.

    The geometry is ``overlay.renderer.rotation_badge_rect``'s rather than
    a copy of it, because what this is for is checking that what the coach
    paints can still be read back -- and a copy would go on agreeing with
    itself after the renderer moved.
    """
    out = image.copy()
    x, y, w, h = rotation_badge_rect(move, cell, cell)
    yy, xx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
    cy, cx, ry, rx = y + h / 2, x + w / 2, h / 2, w / 2
    inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    out[inside] = np.array(HINT, dtype=np.uint8)
    return out


def paint_opaque(image: np.ndarray, cells: list[tuple[int, int]], cell: int = CELL) -> np.ndarray:
    """Mark cells with an OPAQUE patch of our own colour, as a badge is.

    Not a ring, so ``own_paint_states`` reads it as our drawing rather
    than as our translucent fill: the cell is SKIPPED, whatever the game
    drew in it. That is what the rotation badge is, and what a hint cell
    whose outline is cut short by a board rectangle a few pixels off
    becomes -- the widening commit fba8c0f made deliberately, erring
    toward losing a cell we cannot read over inventing one we can.
    """
    out = image.copy()
    for r, c in cells:
        cy, cx = int((r + 0.5) * cell), int((c + 0.5) * cell)
        radius = max(2, round(0.2 * cell))
        yy, xx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = np.array(HINT, dtype=np.uint8)
    return out

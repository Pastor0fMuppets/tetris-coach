"""Recording which cells ``vision.grid`` takes out of a frame, and why.

Two rules can name a third rendering level and remove it from the
occupancy split: ``_own_paint_layer``, which recognizes the coach's OWN
hint overlay coming back round in its own capture, and ``_ghost_layer``,
which names a game-drawn landing preview from structure alone. Both
DELETE cells, so what the session replays pin is not "those cells read
empty" — a faint cell lands in the empty class under the ordinary split
too — but which cells a rule took responsibility for, and which rule it
was.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from tetris_coach.vision import grid as vision_grid

# ``overlay.renderer.draw_hint`` strokes each hint cell's outline with
# ``HintStyle.color`` at FULL opacity, so a captured frame that contains
# the coach's own overlay contains that exact RGB value and a frame that
# does not, does not. This is the ground truth the classifier's rule is
# checked against: the rule works off the translucent FILL composited
# onto the background, which is a different measurement of the same paint.
HINT_RGB = (0x00, 0xE5, 0xFF)


def pen_stroked_cells(path: Path, rows: int, cols: int = 10) -> frozenset[tuple[int, int]]:
    """Cells of ``path`` whose pixels contain the hint pen's own color.

    Read off the raw PNG (which is RGB; the pipeline hands the engine
    BGR), with a small channel tolerance for the renderer's antialiasing
    and a floor of 20 px so a stray blended pixel is not a cell. This
    includes the ROTATION BADGE, which is painted in the same color but
    opaque and one cell above the widget — the classifier cannot name
    that one, which is exactly why it refuses those frames whole.
    """
    image = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    height, width, _ = image.shape
    hit = np.abs(image - np.array(HINT_RGB)).max(axis=-1) <= 12
    found = set()
    for row in range(rows):
        for col in range(cols):
            y0, y1 = int(row * height / rows), int((row + 1) * height / rows)
            x0, x1 = int(col * width / cols), int((col + 1) * width / cols)
            if int(hit[y0:y1, x0:x1].sum()) > 20:
                found.add((row, col))
    return frozenset(found)


class NamedLayer(NamedTuple):
    """One layer a rule removed from one frame."""

    rule: str  # "own_paint" or "ghost"
    cells: frozenset[tuple[int, int]]
    in_flight: frozenset[str]  # whole tetrominoes airborne on that frame


@contextmanager
def layers_named():  # type: ignore[no-untyped-def]
    """Yield a list that fills with a :class:`NamedLayer` per named layer."""
    seen: list[NamedLayer] = []
    real_paint = vision_grid._own_paint_layer
    real_ghost = vision_grid._ghost_layer

    def cells_of(layer):  # type: ignore[no-untyped-def]
        return frozenset((int(r), int(c)) for r, c in zip(*np.nonzero(layer), strict=True))

    def record_paint(colors, background, solid, unobservable, paint):  # type: ignore[no-untyped-def]
        layer = real_paint(colors, background, solid, unobservable, paint)
        if layer is not None:
            seen.append(
                NamedLayer(
                    "own_paint",
                    cells_of(layer),
                    frozenset(vision_grid._pieces_in_flight(solid, unobservable)),
                )
            )
        return layer

    def record_ghost(scores, unobservable):  # type: ignore[no-untyped-def]
        layer = real_ghost(scores, unobservable)
        if layer is not None:
            solid = scores >= vision_grid.MIN_SPREAD
            for cell in unobservable:
                solid[cell] = False
            seen.append(
                NamedLayer(
                    "ghost",
                    cells_of(layer),
                    frozenset(vision_grid._pieces_in_flight(solid, unobservable)),
                )
            )
        return layer

    vision_grid._own_paint_layer = record_paint  # type: ignore[assignment]
    vision_grid._ghost_layer = record_ghost  # type: ignore[assignment]
    try:
        yield seen
    finally:
        vision_grid._own_paint_layer = real_paint  # type: ignore[assignment]
        vision_grid._ghost_layer = real_ghost  # type: ignore[assignment]

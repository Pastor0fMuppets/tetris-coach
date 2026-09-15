"""Unit tests for the colour segmentation the parallel tracker reads from.

What a cell's colour is (:mod:`~tetris_coach.vision.colour_palette`), what
is this tool's own paint rather than the game's, and what the NEXT box holds
(:mod:`~tetris_coach.vision.colour_preview`).
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris_coach.vision.colour_palette import (
    CLEAN,
    EMPTY,
    OURS,
    PAINTED,
    Palette,
    board_colours,
    observable_mask,
    own_paint_states,
    piece_from_cells,
)
from tetris_coach.vision.colour_preview import identify_preview

from .colour_frames import (
    BLUE_I,
    CELL,
    COLS,
    GREEN_O,
    HINT,
    PALE_T,
    ROWS,
    WHITE,
    blend,
    paint_hint,
    preview,
    render,
    tracker,
)

# -- the palette ------------------------------------------------------


def test_a_paler_shade_of_a_colour_is_the_same_class() -> None:
    """The NEXT box draws a piece at ~87% opacity; that is the same piece."""
    palette = Palette()
    palette.update_background(
        board_colours(render({}), ROWS, COLS), observable_mask(ROWS, COLS, None)
    )
    assert palette.background is not None
    solid = palette.intern(np.array(BLUE_I) - palette.background)
    faded = palette.intern(np.array(blend(BLUE_I, 0.87)) - palette.background)
    assert solid == faded
    assert len(palette.classes) == 1


def test_different_pieces_are_different_classes() -> None:
    palette = Palette()
    palette.update_background(
        board_colours(render({}), ROWS, COLS), observable_mask(ROWS, COLS, None)
    )
    assert palette.background is not None
    seen = {palette.intern(np.array(c) - palette.background) for c in (BLUE_I, GREEN_O, PALE_T)}
    assert len(seen) == 3


def test_piece_from_cells_names_only_complete_tetrominoes() -> None:
    assert piece_from_cells(frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})) == "O"
    assert piece_from_cells(frozenset({(5, 1), (6, 0), (6, 1), (6, 2)})) == "T"
    assert piece_from_cells(frozenset({(0, 0), (0, 1)})) is None
    assert piece_from_cells(frozenset({(0, 0), (0, 1), (0, 2), (5, 5)})) is None


def test_our_own_hint_paint_over_the_board_is_not_content() -> None:
    """The hint we drew last frame comes back in this one. It is not a piece."""
    image = paint_hint(render({}), [(10, 0), (10, 1), (11, 0), (11, 1)])
    states = own_paint_states(image, ROWS, COLS)
    assert states[10, 0] == PAINTED and states[5, 5] == CLEAN
    report = tracker().update(image)
    assert report.falling is None
    assert report.stack_rows == (0,) * ROWS


def test_our_own_hint_paint_over_a_piece_keeps_the_piece() -> None:
    """The fill is undone exactly, so what the game drew still reads."""
    board = {(11, c): BLUE_I for c in range(4)}
    image = paint_hint(render(board), [(11, c) for c in range(4)])
    report = tracker().update(image)
    assert report.stack_rows[11] == 0b1111
    assert report.falling is None


def test_our_own_rotation_badge_is_not_content() -> None:
    """A disc of pure hint colour is our drawing, not an O."""
    image = render({})
    hint = np.array(HINT, dtype=np.uint8)
    # renderer._draw_rotation_badge: an ellipse 0.6 x 0.55 of a cell, hung on
    # the left edge of the cell above the hint's top-left corner.
    yy, xx = np.mgrid[0 : ROWS * CELL, 0 : COLS * CELL]
    cy, cx = (5 + 0.275) * CELL, (3 + 0.3) * CELL
    ry, rx = 0.275 * CELL, 0.3 * CELL
    image[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = hint
    assert own_paint_states(image, ROWS, COLS)[5, 3] == OURS
    report = tracker().update(image)
    assert report.falling is None and report.stack_rows == (0,) * ROWS


# -- the NEXT box -----------------------------------------------------


@pytest.mark.parametrize(
    ("cells", "piece"),
    [
        ([(0, 0), (0, 1), (0, 2), (0, 3)], "I"),
        ([(0, 0), (1, 0), (2, 0), (3, 0)], "I"),
        ([(0, 0), (0, 1), (1, 0), (1, 1)], "O"),
        ([(0, 1), (1, 0), (1, 1), (1, 2)], "T"),
        ([(0, 1), (0, 2), (1, 0), (1, 1)], "S"),
        ([(0, 0), (0, 1), (1, 1), (1, 2)], "Z"),
        ([(0, 0), (1, 0), (1, 1), (1, 2)], "J"),
        ([(0, 2), (1, 0), (1, 1), (1, 2)], "L"),
    ],
)
def test_the_preview_names_every_piece_and_returns_its_colour(
    cells: list[tuple[int, int]], piece: str
) -> None:
    reading = identify_preview(preview(cells, GREEN_O))
    assert reading is not None
    assert reading.piece == piece
    assert np.allclose(reading.colour, np.array(GREEN_O), atol=1.0)


def test_the_preview_refuses_an_empty_box_and_our_own_paint() -> None:
    empty = np.full((100, 100, 3), np.array(WHITE, dtype=np.uint8), dtype=np.uint8)
    assert identify_preview(empty) is None
    painted = paint_hint(
        np.full((100, 100, 3), np.array(WHITE, dtype=np.uint8), dtype=np.uint8),
        [(1, 1), (1, 2), (2, 1), (2, 2)],
        cell=20,
    )
    assert identify_preview(painted) is None


def test_the_palette_classifies_every_cell_of_a_frame() -> None:
    palette = Palette()
    image = render({(0, 0): BLUE_I, (11, 9): GREEN_O})
    observable = observable_mask(ROWS, COLS, None)
    colours = board_colours(image, ROWS, COLS)
    palette.update_background(colours, observable)
    labels = palette.classify(colours, observable)
    assert labels[0, 0] != EMPTY and labels[11, 9] != EMPTY
    assert labels[0, 0] != labels[11, 9]
    assert int((labels != EMPTY).sum()) == 2

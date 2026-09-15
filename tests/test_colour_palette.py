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


def test_a_disc_of_hint_colour_in_a_patch_is_our_drawing() -> None:
    """Hint colour ACROSS a cell's patch is ours, and the cell is unreadable.

    This is the rule, on the shape the rotation badge used to have: an
    ellipse 0.6 x 0.55 of a cell, hung on the left edge of the cell above
    the hint's top-left corner. The badge is not drawn there any more --
    it deleted whatever the game had in that cell, and the piece passes
    through it (see
    ``test_engine_colour.test_the_rotation_badge_no_longer_deletes_the_piece``)
    -- but the rule stays, because the cell really is unreadable when
    something opaque covers the middle of it.
    """
    image = render({})
    hint = np.array(HINT, dtype=np.uint8)
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


def test_the_palette_is_a_global_one_way_memory_and_the_docstring_says_so() -> None:
    """The design's claim is about memory, so what memory exists is pinned.

    "A misread frame costs exactly that frame" was the headline and it was
    not true of this object: classify() interns from every content cell of
    every frame, and three of its changes cannot be undone. That is now
    written down rather than claimed away, and asserted here so it stays
    true of the code and not only of the prose.
    """
    palette = Palette(paint=None)
    palette.background = np.zeros(3)
    bright = np.array([0.0, 0.0, 200.0])
    index = palette.intern(bright)
    palette.witness(index, "I")
    assert palette.classes[index].peak == 200.0

    # A class is never removed, and a peak never falls.
    palette.intern(np.array([0.0, 0.0, 50.0]))
    assert len(palette.classes) == 1, "the dimmer shade joins the same ray"
    assert palette.classes[index].peak == 200.0
    palette.intern(np.array([200.0, 0.0, 0.0]))
    assert len(palette.classes) == 2, "and a new ray is a new class, for good"

    # A retired name never returns.
    palette.witness(index, "O")
    assert palette.piece_of(index) is None
    palette.witness(index, "I")
    palette.name(index, "I")
    assert palette.piece_of(index) is None

    # But the peak no longer decides what is content, which is what made
    # the one-way part dangerous: both shades of the one ray are content.
    labels = palette.classify(
        np.array([[bright, np.array([0.0, 0.0, 50.0])]], dtype=np.float32),
        np.ones((1, 2), dtype=bool),
    )
    assert labels.tolist() == [[index, index]]


def test_a_palette_can_be_put_back_exactly_as_it_was() -> None:
    """Reading a frame writes here, so a frame can have to be taken back.

    The tracker's gate cannot ask whether a frame is board-SHAPED until the
    cells are labelled, and labelling them interns classes and raises peaks
    -- both one-way. Rolling back is what keeps "a frame that is not a
    board writes nothing" true for the half of the gate that has to look
    first.
    """
    palette = Palette(paint=None)
    palette.update_background(
        np.array([[[250.0, 250.0, 250.0]]], dtype=np.float32), np.ones((1, 1), dtype=bool)
    )
    blue = palette.intern(np.array([-40.0, -200.0, -200.0]))
    palette.name(blue, "I")
    mark = palette.checkpoint()

    junk = palette.intern(np.array([120.0, -30.0, 90.0]))
    palette.intern(np.array([-80.0, -400.0, -400.0]))  # a brighter sighting: raises the peak
    palette.witness(blue, "O")  # and retires the name
    assert junk != blue and len(palette.classes) == 2
    assert palette.piece_of(blue) is None and palette.classes[blue].peak > 0.0
    peak = palette.classes[blue].peak

    palette.restore(mark)
    assert len(palette.classes) == 1
    assert palette.piece_of(blue) == "I"
    assert palette.classes[blue].peak < peak
    assert palette.background is not None

    # Restoring twice from the same mark is the same thing: a checkpoint is
    # a copy, not a view of the palette it came from.
    palette.intern(np.array([120.0, -30.0, 90.0]))
    palette.restore(mark)
    assert len(palette.classes) == 1
